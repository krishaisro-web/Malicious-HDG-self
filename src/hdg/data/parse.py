"""
Malicious-HDG: Streaming Dataset Parser & Multi-Source Normalization Engine.

Purpose & Workflow:
-------------------
Provides memory-efficient streaming parsing and leak-free normalization for raw Zenodo
DomainRadar v2 JSON records (up to 930,000 domains across 3 raw feeds: malware, Cisco Umbrella,
and CESNET ground truth):
1. Streaming Ingestion:
   - Uses `ijson` iterative JSON streaming (`ijson.items(f, "item")`) to process multi-gigabyte
     raw JSON files without reading entire files into memory.
   - Enforces a <1% malformed record tolerance threshold (fails closed on dirty or truncated dumps).
2. Leak-Free Normalization:
   - Resolves effective second-level domains (e2LDs) via `tldextract` without relying on dynamic
     Public Suffix List network fetches during test execution.
   - Resolves timestamps via strict fallback: `sourced_on` -> `evaluated_on`. Excludes records
     lacking valid timestamps (never imputes artificial min-dates to prevent temporal leakage).
3. Infrastructure Key Resolution:
   - Certificates: Extracts leaf certs using `is_root=False`. Computes dual keys:
       * Common Name (CN) key: hash(CN, org, country, valid_start, valid_end)
       * Co-issuance key: hash(org, country, valid_start, valid_end, valid_len) [no CN]
   - Autonomous Systems: Traverses candidate keys (`asn`, `autonomous_system_number`, `number`).
   - Registrars: Traverses entity records in RDAP objects (`name`, `handle`, `organization`).
4. Cross-Class & Intra-Class Deduplication:
   - Intra-class: Deduplicates by domain unit, retaining the chronologically earliest observation.
   - Cross-class: Drops domains present in both clean benign and malware feeds to eliminate label ambiguity.
5. Cohort Sampling:
   - Time-matched: Matches benign domain counts per month against malware volume over overlapping months.
   - Natural Scale: Preserves natural ~8.24:1 class imbalance when `benign_ratio is None`.
"""

from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import ijson
import numpy as np
import pandas as pd
import tldextract

from src.hdg.data.profiler import parse_mongo_date
from src.hdg.data.features import (
    compute_longest_consonant_run,
    compute_shannon_entropy,
    is_multitenant_public_suffix_apex,
)

logger = logging.getLogger(__name__)


def extract_leaf_certificate_details(tls_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Finds the first non-root certificate (leaf) using is_root flag (not position).
    Extracts leaf keys plus intrinsic certificate attributes for multi-feature node representation.
    Note: Zenodo certificates have no fingerprint/serial, so identical-metadata certs collide.
    """
    empty: Dict[str, Any] = {
        "leaf_cert_cn": None,
        "leaf_cert_coissue": None,
        "cert_valid_len": None,
        "cert_org": None,
        "cert_country": None,
        "cert_ext_count": 0,
        "cert_is_self_signed": False,
    }
    if not isinstance(tls_obj, dict):
        return empty

    certs = tls_obj.get("certificates")
    if not isinstance(certs, list):
        return empty

    leaf_cert = None
    for c in certs:
        if isinstance(c, dict) and not c.get("is_root", False):
            leaf_cert = c
            break

    if leaf_cert is None:
        return empty

    cn = str(leaf_cert.get("common_name") or "").strip()
    org = str(leaf_cert.get("organization") or "").strip()
    country = str(leaf_cert.get("country") or "").strip()
    v_start = parse_mongo_date(leaf_cert.get("validity_start"))
    v_end = parse_mongo_date(leaf_cert.get("validity_end"))
    v_start_str = v_start.isoformat() if v_start else ""
    v_end_str = v_end.isoformat() if v_end else ""
    valid_len = leaf_cert.get("valid_len")
    val_len_int = int(valid_len) if valid_len is not None else None

    ext_count_raw = leaf_cert.get("extension_count")
    if ext_count_raw is not None:
        try:
            ext_count = int(ext_count_raw)
        except (ValueError, TypeError):
            ext_count = 0
    elif isinstance(leaf_cert.get("extensions"), list):
        ext_count = len(leaf_cert["extensions"])
    else:
        ext_count = 0

    issuer_val = str(leaf_cert.get("issuer") or "").strip().lower()
    subject_val = str(leaf_cert.get("subject") or "").strip().lower()
    is_self_signed = bool(
        (issuer_val and subject_val and issuer_val == subject_val)
        or (cn and org and cn.lower() == org.lower())
    )

    # Key 1: with common_name (standard leaf cert identifier)
    meta_cn = f"{cn}|{org}|{country}|{v_start_str}|{v_end_str}"
    cert_hash_cn = hashlib.sha256(meta_cn.encode("utf-8")).hexdigest()[:16]

    # Key 2: co-issuance key (omits common_name to capture shared infrastructure / batch issuance)
    meta_coissue = f"{org}|{country}|{v_start_str}|{v_end_str}|{val_len_int}"
    cert_hash_coissue = hashlib.sha256(meta_coissue.encode("utf-8")).hexdigest()[:16]

    return {
        "leaf_cert_cn": cert_hash_cn,
        "leaf_cert_coissue": cert_hash_coissue,
        "cert_valid_len": val_len_int,
        "cert_org": org if org else None,
        "cert_country": country if country else None,
        "cert_ext_count": ext_count,
        "cert_is_self_signed": is_self_signed,
    }


def extract_leaf_certificate(tls_obj: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Finds the first non-root certificate (leaf) using is_root flag (not position).
    Returns (leaf_cert_key_cn, leaf_cert_key_coissue, validity_length_days).
    """
    details = extract_leaf_certificate_details(tls_obj)
    return details["leaf_cert_cn"], details["leaf_cert_coissue"], details["cert_valid_len"]


def resolve_asn(
    asn_obj: Optional[Dict[str, Any]],
    candidate_keys: List[str],
    asn_key_counter: Counter
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Resolves ASN number, organization, and optional BGP prefix/network from candidate keys.
    Handles both Zenodo DomainRadar v2 keys (as_org, network_address, prefix_len) and legacy keys.
    """
    if not isinstance(asn_obj, dict):
        return None, None, None

    asn_num: Optional[int] = None
    for k in candidate_keys:
        if k in asn_obj and asn_obj[k] is not None:
            try:
                asn_num = int(asn_obj[k])
                asn_key_counter[k] += 1
                break
            except (ValueError, TypeError):
                continue

    org = str(asn_obj.get("as_org") or asn_obj.get("organization") or "").strip()
    net_addr = str(asn_obj.get("network_address") or asn_obj.get("network") or "").strip()
    plen = asn_obj.get("prefix_len")
    if net_addr and plen is not None and "/" not in net_addr:
        prefix = f"{net_addr}/{plen}"
    else:
        prefix = net_addr if net_addr else None
    return asn_num, (org if org else None), prefix


def resolve_registrar_details(
    rdap_obj: Optional[Dict[str, Any]],
    candidate_keys: List[str],
    registrar_key_counter: Counter
) -> Dict[str, Any]:
    """
    Resolves registrar name and contact/metadata flags from rdap.entities.registrar[].
    """
    empty = {
        "registrar": None,
        "reg_has_whois": False,
        "reg_has_email": False,
        "reg_has_url": False,
        "reg_has_tel": False,
    }
    if not isinstance(rdap_obj, dict):
        return empty

    entities = rdap_obj.get("entities")
    if not isinstance(entities, dict):
        return empty

    has_abuse_email = False
    abuse_list = entities.get("abuse")
    if isinstance(abuse_list, list):
        has_abuse_email = any(isinstance(a, dict) and bool(a.get("email")) for a in abuse_list)

    reg_list = entities.get("registrar")
    if not isinstance(reg_list, list):
        return empty

    for item in reg_list:
        if isinstance(item, dict):
            reg_val: Optional[str] = None
            for k in candidate_keys:
                if k in item and item[k]:
                    val = str(item[k]).strip()
                    if val:
                        registrar_key_counter[k] += 1
                        reg_val = val
                        break
            if reg_val is not None:
                return {
                    "registrar": reg_val,
                    "reg_has_whois": bool(item.get("whois_server")),
                    "reg_has_email": bool(item.get("email")) or has_abuse_email,
                    "reg_has_url": bool(item.get("url")),
                    "reg_has_tel": bool(item.get("tel")),
                }
    return empty


def resolve_registrar(
    rdap_obj: Optional[Dict[str, Any]],
    candidate_keys: List[str],
    registrar_key_counter: Counter
) -> Optional[str]:
    """
    Resolves registrar name from candidate entity keys inside rdap.entities.registrar[].
    """
    return resolve_registrar_details(rdap_obj, candidate_keys, registrar_key_counter)["registrar"]


def stream_parse_file(
    file_path: Path,
    label: int,
    default_source: str,
    extractor: tldextract.TLDExtract,
    normalize_to_e2ld: bool,
    asn_candidate_keys: List[str],
    registrar_candidate_keys: List[str],
    asn_key_counter: Counter,
    reg_key_counter: Counter,
    max_records: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Streams a single JSON file, returning parsed records, total seen, excluded (no time), and malformed count.
    Aborts only if malformed records exceed 1% when total seen > 100.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    records: List[Dict[str, Any]] = []
    total_seen = 0
    excluded_no_time = 0
    malformed_count = 0

    with open(file_path, "rb") as f:
        stream = ijson.items(f, "item", use_float=True)
        for r in stream:
            if not isinstance(r, dict):
                malformed_count += 1
                continue
            total_seen += 1

            try:
                raw_domain = (r.get("domain_name") or "").strip().lower()
                if not raw_domain:
                    continue

                # Time resolution: t_source fallback t_eval; NEVER min-date
                t_src = parse_mongo_date(r.get("sourced_on"))
                t_eval = parse_mongo_date(r.get("evaluated_on"))
                t = t_src if t_src is not None else t_eval

                if t is None:
                    excluded_no_time += 1
                    continue

                # e2LD extraction
                ext_res = extractor(raw_domain)
                if ext_res.domain and ext_res.suffix:
                    e2ld = f"{ext_res.domain}.{ext_res.suffix}".lower()
                else:
                    e2ld = raw_domain

                has_subdomain = bool(ext_res.subdomain)
                domain_unit = e2ld if normalize_to_e2ld else raw_domain
                if normalize_to_e2ld:
                    domain_body = (ext_res.domain or domain_unit).lower()
                else:
                    domain_body = (
                        f"{ext_res.subdomain}.{ext_res.domain}".lower()
                        if ext_res.subdomain and ext_res.domain
                        else (ext_res.domain or domain_unit).lower()
                    )

                # Source and family
                source = str(r.get("source") or default_source)
                # Malware family key conflict: read both r.get("malware_type") or r.get("malware")
                fam = r.get("malware_type") or r.get("malware")
                family = str(fam) if (label == 1 and fam is not None) else None

                # RDAP
                rdap = r.get("rdap")
                no_rdap = (rdap is None)
                t_reg = parse_mongo_date(rdap.get("registration_date")) if isinstance(rdap, dict) else None
                reg_details = resolve_registrar_details(rdap, registrar_candidate_keys, reg_key_counter)
                registrar = reg_details["registrar"]

                # Nameservers: union of dns.NS and rdap.nameservers
                ns_set: Set[str] = set()
                dns = r.get("dns")
                if isinstance(dns, dict):
                    ns_obj = dns.get("NS")
                    if isinstance(ns_obj, dict):
                        for ns_name in ns_obj.keys():
                            ns_clean = str(ns_name).strip().lower().rstrip(".")
                            if ns_clean:
                                ns_set.add(ns_clean)

                if isinstance(rdap, dict):
                    rdap_ns = rdap.get("nameservers")
                    if isinstance(rdap_ns, list):
                        for ns_item in rdap_ns:
                            ns_clean = str(ns_item).strip().lower().rstrip(".")
                            if ns_clean:
                                ns_set.add(ns_clean)

                # DNS Flags & Cardinality
                mx_hosts: List[str] = []
                cname_targets: List[str] = []
                if isinstance(dns, dict):
                    a_recs = dns.get("A")
                    aaaa_recs = dns.get("AAAA")
                    a_count = len(a_recs) if isinstance(a_recs, (list, dict)) else 0
                    aaaa_count = len(aaaa_recs) if isinstance(aaaa_recs, (list, dict)) else 0
                    has_a = bool(a_recs)
                    has_aaaa = bool(aaaa_recs)
                    no_ip = not (has_a or has_aaaa)
                    mx_dict = dns.get("MX")
                    if isinstance(mx_dict, dict):
                        mx_count = len(mx_dict)
                        mx_hosts = sorted([str(k).strip().lower().rstrip(".") for k in mx_dict.keys() if k])
                    elif isinstance(mx_dict, list):
                        mx_count = len(mx_dict)
                    else:
                        mx_count = 0
                    cname_val = dns.get("CNAME")
                    cname_present = bool(cname_val)
                    if isinstance(cname_val, dict):
                        cname_targets = sorted([str(k).strip().lower().rstrip(".") for k in cname_val.keys() if k])
                    elif isinstance(cname_val, list):
                        cname_targets = sorted([str(x).strip().lower().rstrip(".") for x in cname_val if x])
                    elif isinstance(cname_val, str) and cname_val.strip():
                        cname_targets = [cname_val.strip().lower().rstrip(".")]
                    soa_obj = dns.get("SOA")
                    soa_present = isinstance(soa_obj, dict) and bool(soa_obj)
                    if no_ip and soa_present and soa_obj.get("primary_ns"):
                        soa_ns = str(soa_obj["primary_ns"]).strip().lower().rstrip(".")
                        if soa_ns:
                            ns_set.add(soa_ns)
                    txt_recs = dns.get("TXT")
                    txt_count = len(txt_recs) if isinstance(txt_recs, (list, dict)) else 0
                    remarks = dns.get("remarks") if isinstance(dns.get("remarks"), dict) else {}
                    has_spf = bool(remarks.get("has_spf", False))
                    has_dmarc = bool(remarks.get("has_dmarc", False))
                    has_dkim = bool(remarks.get("has_dkim", False))
                    dnssec = bool(dns.get("dnssec"))
                    ttls = dns.get("ttls") if isinstance(dns.get("ttls"), dict) else {}
                else:
                    no_ip = True
                    a_count = 0
                    aaaa_count = 0
                    mx_count = 0
                    cname_present = False
                    soa_present = False
                    txt_count = 0
                    has_spf = False
                    has_dmarc = False
                    has_dkim = False
                    dnssec = False
                    ttls = {}

                # TLS & Certificates (both CN and co-issuance keys + intrinsic leaf metadata)
                tls = r.get("tls")
                no_tls = (tls is None)
                tls_proto = str(tls.get("protocol")) if (isinstance(tls, dict) and tls.get("protocol")) else None
                tls_cipher = str(tls.get("cipher")) if (isinstance(tls, dict) and tls.get("cipher")) else None
                certs = tls.get("certificates") if isinstance(tls, dict) else None
                cert_count = len(certs) if isinstance(certs, list) else 0
                cert_details = extract_leaf_certificate_details(tls)
                leaf_cert_cn = cert_details["leaf_cert_cn"]
                leaf_cert_coissue = cert_details["leaf_cert_coissue"]
                cert_valid_len = cert_details["cert_valid_len"]

                # IP Data: collect IPs, ASN, Geo, and optional BGP prefix
                resolved_ips: List[str] = []
                domain_asns: Set[int] = set()
                ip_records: List[Dict[str, Any]] = []

                ip_data = r.get("ip_data")
                if isinstance(ip_data, list):
                    for entry in ip_data:
                        if not isinstance(entry, dict):
                            continue
                        ip_val = str(entry.get("ip") or "").strip()
                        from_rec = str(entry.get("from_record") or "").upper()
                        if not ip_val:
                            continue

                        # Filter only A/AAAA IPs for direct domain-ip resolution edges
                        if from_rec in {"A", "AAAA"}:
                            resolved_ips.append(ip_val)

                        asn_obj = entry.get("asn")
                        asn_num, asn_org, bgp_prefix = resolve_asn(asn_obj, asn_candidate_keys, asn_key_counter)
                        if asn_num is not None:
                            domain_asns.add(asn_num)

                        prefix_len_val: Optional[int] = None
                        net_addr_val: Optional[str] = None
                        if isinstance(asn_obj, dict):
                            if asn_obj.get("prefix_len") is not None:
                                try:
                                    prefix_len_val = int(asn_obj.get("prefix_len"))
                                except (ValueError, TypeError):
                                    prefix_len_val = None
                            raw_net = asn_obj.get("network_address") or asn_obj.get("network")
                            if raw_net:
                                net_addr_val = str(raw_net).strip()
                        if prefix_len_val is None and bgp_prefix and "/" in bgp_prefix:
                            try:
                                prefix_len_val = int(bgp_prefix.split("/", 1)[1])
                            except (ValueError, TypeError):
                                prefix_len_val = None

                        geo_obj = entry.get("geo")
                        if isinstance(geo_obj, dict):
                            country = str(geo_obj.get("country")) if geo_obj.get("country") else None
                            country_code = str(geo_obj.get("country_code")) if geo_obj.get("country_code") else None
                            geo_isp = str(geo_obj.get("isp")) if geo_obj.get("isp") else None
                            geo_org = str(geo_obj.get("org")) if geo_obj.get("org") else None
                            try:
                                lat_val = float(geo_obj["latitude"]) if geo_obj.get("latitude") is not None else None
                            except (ValueError, TypeError):
                                lat_val = None
                            try:
                                lon_val = float(geo_obj["longitude"]) if geo_obj.get("longitude") is not None else None
                            except (ValueError, TypeError):
                                lon_val = None
                        else:
                            country = None
                            country_code = None
                            geo_isp = None
                            geo_org = None
                            lat_val = None
                            lon_val = None

                        ip_records.append({
                            "ip": ip_val,
                            "from_record": from_rec,
                            "asn": asn_num,
                            "asn_org": asn_org,
                            "country": country,
                            "country_code": country_code,
                            "isp": geo_isp,
                            "org": geo_org,
                            "latitude": lat_val,
                            "longitude": lon_val,
                            "prefix_len": prefix_len_val,
                            "network_address": net_addr_val,
                            "prefix": bgp_prefix
                        })

                # Also check related_ips in dns.NS and dormant dns.MX when no_ip to retain infra relations
                ns_ip_pairs: List[Tuple[str, str]] = []
                mx_ip_pairs: List[Tuple[str, str]] = []
                if isinstance(dns, dict):
                    if isinstance(dns.get("NS"), dict):
                        for ns_host, ns_val in dns["NS"].items():
                            ns_clean = str(ns_host).strip().lower().rstrip(".")
                            if isinstance(ns_val, dict) and isinstance(ns_val.get("related_ips"), list):
                                for rel in ns_val["related_ips"]:
                                    if isinstance(rel, dict) and rel.get("value"):
                                        ns_ip_pairs.append((ns_clean, str(rel["value"]).strip()))
                    if no_ip and isinstance(dns.get("MX"), dict):
                        for mx_host, mx_val in dns["MX"].items():
                            mx_clean = str(mx_host).strip().lower().rstrip(".")
                            if isinstance(mx_val, dict) and isinstance(mx_val.get("related_ips"), list):
                                for rel in mx_val["related_ips"]:
                                    if isinstance(rel, dict) and rel.get("value"):
                                        mx_ip_pairs.append((mx_clean, str(rel["value"]).strip()))

                primary_asn = sorted(list(domain_asns))[0] if domain_asns else None
                sub_str = (ext_res.subdomain or "").strip().lower()
                subdomain_count = len([p for p in sub_str.split(".") if p]) if sub_str else 0
                tld_str = (ext_res.suffix or (e2ld.rsplit(".", 1)[-1] if "." in e2ld else "")).lower()

                rec = {
                    "domain": domain_unit,
                    "raw_domain": raw_domain,
                    "e2LD": e2ld,
                    "apex_domain": e2ld,
                    "tld": tld_str,
                    "subdomain_label": sub_str,
                    "subdomain_count": subdomain_count,
                    "domain_body": domain_body,
                    "has_subdomain": has_subdomain,
                    "label": label,
                    "source": source,
                    "family": family,
                    "t": t.isoformat(),
                    "t_month": t.strftime("%Y-%m"),
                    "t_source": t_src.isoformat() if t_src else None,
                    "t_eval": t_eval.isoformat() if t_eval else None,
                    "t_reg": t_reg.isoformat() if t_reg else None,
                    "no_ip": no_ip,
                    "no_rdap": no_rdap,
                    "no_tls": no_tls,
                    "a_count": a_count,
                    "aaaa_count": aaaa_count,
                    "mx_count": mx_count,
                    "soa_present": soa_present,
                    "cname_present": cname_present,
                    "txt_count": txt_count,
                    "mx_hosts": mx_hosts,
                    "cname_targets": cname_targets,
                    "mx_ip_pairs": mx_ip_pairs,
                    "has_spf": has_spf,
                    "has_dmarc": has_dmarc,
                    "has_dkim": has_dkim,
                    "dnssec": dnssec,
                    "ttl_a": int(ttls.get("A", 0)) if isinstance(ttls, dict) and ttls.get("A") is not None else 0,
                    "ttl_ns": int(ttls.get("NS", 0)) if isinstance(ttls, dict) and ttls.get("NS") is not None else 0,
                    "nameservers": sorted(list(ns_set)),
                    "registrar": registrar,
                    "reg_has_whois": reg_details["reg_has_whois"],
                    "reg_has_email": reg_details["reg_has_email"],
                    "reg_has_url": reg_details["reg_has_url"],
                    "reg_has_tel": reg_details["reg_has_tel"],
                    "tls_protocol": tls_proto,
                    "tls_cipher": tls_cipher,
                    "tls_cert_count": cert_count,
                    "leaf_cert_key": leaf_cert_cn,
                    "leaf_cert_key_cn": leaf_cert_cn,
                    "leaf_cert_key_coissue": leaf_cert_coissue,
                    "cert_valid_len": cert_valid_len,
                    "cert_org": cert_details["cert_org"],
                    "cert_country": cert_details["cert_country"],
                    "cert_ext_count": cert_details["cert_ext_count"],
                    "cert_is_self_signed": cert_details["cert_is_self_signed"],
                    "primary_asn": primary_asn,
                    "asns": sorted(list(domain_asns)),
                    "resolved_ips": sorted(list(set(resolved_ips))),
                    "ip_records": ip_records,
                    "ns_ip_pairs": ns_ip_pairs
                }
                records.append(rec)

                if max_records is not None and len(records) >= max_records:
                    break

            except Exception as e:
                malformed_count += 1
                if total_seen > 100 and (malformed_count / total_seen) > 0.01:
                    raise ValueError(
                        f"Malformed records exceeded 1% threshold in {file_path.name}: "
                        f"{malformed_count}/{total_seen} malformed records. Last error: {e}"
                    )
                continue

    if malformed_count > 0:
        logger.warning(f"File {file_path.name} had {malformed_count} malformed records (out of {total_seen}).")

    return records, total_seen, excluded_no_time, malformed_count


def export_dual_fqdn_apex_tables(
    primary_domains: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exports RFC-001 normalized dictionary-encoded dual Parquet tables:
    1. `domains_apex.parquet` (unique apex entities, fan-out counts, WHOIS metadata)
    2. `domains_fqdn.parquet` (FQDN records with integer foreign key `apex_id` + flat lexical cols)
    Also attaches `apex_id`, `subdomain_fanout_count`, and `is_public_suffix` in-place on `primary_domains`.
    """
    apex_to_id: Dict[str, int] = {}
    apex_meta: Dict[str, Dict[str, Any]] = {}
    apex_fanout: Counter = Counter()

    for r in primary_domains:
        apex_dom = str(r.get("apex_domain") or r.get("e2LD") or r.get("domain") or "").strip().lower()
        apex_fanout[apex_dom] += 1
        if apex_dom not in apex_to_id:
            apex_to_id[apex_dom] = len(apex_to_id)
            apex_meta[apex_dom] = {
                "apex_id": apex_to_id[apex_dom],
                "apex_domain": apex_dom,
                "tld": str(r.get("tld") or (apex_dom.rsplit(".", 1)[-1] if "." in apex_dom else "")),
                "registration_date": r.get("t_reg"),
                "whois_registrar": r.get("registrar"),
                "whois_organization": (r.get("ip_records") or [{}])[0].get("org") if r.get("ip_records") else None,
                "is_public_suffix": bool(is_multitenant_public_suffix_apex(apex_dom)),
            }
        else:
            m = apex_meta[apex_dom]
            if not m.get("registration_date") and r.get("t_reg"):
                m["registration_date"] = r.get("t_reg")
            if not m.get("whois_registrar") and r.get("registrar"):
                m["whois_registrar"] = r.get("registrar")

    apex_rows: List[Dict[str, Any]] = []
    for apex_dom, m in apex_meta.items():
        fan = int(apex_fanout[apex_dom])
        m["subdomain_fanout_count"] = fan
        apex_rows.append(dict(m))

    fqdn_rows: List[Dict[str, Any]] = []
    for d_id, r in enumerate(primary_domains):
        apex_dom = str(r.get("apex_domain") or r.get("e2LD") or r.get("domain") or "").strip().lower()
        a_id = apex_to_id[apex_dom]
        a_info = apex_meta[apex_dom]
        r["apex_domain"] = apex_dom
        r["apex_id"] = a_id
        r["subdomain_fanout_count"] = a_info["subdomain_fanout_count"]
        r["is_public_suffix"] = a_info["is_public_suffix"]

        fqdn_str = str(r.get("raw_domain") or r.get("domain") or "").strip().lower()
        body = str(r.get("domain_body") or fqdn_str)
        sub_prefix = str(r.get("subdomain_label") or "")
        n_b = max(1, len(body))
        fqdn_rows.append({
            "domain_id": int(d_id),
            "fqdn": fqdn_str,
            "apex_id": int(a_id),
            "label": int(r["label"]),
            "source": str(r.get("source") or ""),
            "malware_family": r.get("family"),
            "sourced_on": r.get("t_source"),
            "evaluated_on": r.get("t_eval"),
            "f_length": int(len(body)),
            "f_entropy": float(compute_shannon_entropy(body)),
            "f_digit_ratio": float(sum(ch.isdigit() for ch in body) / n_b),
            "f_vowel_ratio": float(sum(ch in "aeiou" for ch in body.lower()) / n_b),
            "f_longest_consonant_run": int(compute_longest_consonant_run(body)),
            "f_subdomain_count": int(r.get("subdomain_count") or 0),
            "f_prefix_brand_entropy": float(compute_shannon_entropy(sub_prefix)) if sub_prefix else 0.0,
        })

    df_apex = pd.DataFrame(apex_rows)
    df_fqdn = pd.DataFrame(fqdn_rows)
    df_apex.to_parquet(output_dir / "domains_apex.parquet", index=False)
    df_fqdn.to_parquet(output_dir / "domains_fqdn.parquet", index=False)
    return df_fqdn, df_apex


def parse_and_process_dataset(
    raw_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    is_fixture: bool = False,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parses malware, umbrella, and cesnet files.
    Applies cross-class deduplication and time-matched benign sampling over overlapping months.
    Writes data_processed/domains.parquet, domains_timematched.parquet, domains_fqdn.parquet,
    domains_apex.parquet, and summary JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p_cfg = config.get("parsing", {})
    preserve_subdomains = bool(p_cfg.get("preserve_subdomains", False))
    normalize_to_e2ld = False if preserve_subdomains else bool(p_cfg.get("normalize_to_e2LD", True))
    raw_benign_ratio = p_cfg.get("benign_ratio")
    benign_ratio = float(raw_benign_ratio) if raw_benign_ratio is not None else None
    max_per_class = limit if limit is not None else p_cfg.get("max_per_class")
    asn_candidate_keys = p_cfg.get("asn_candidate_keys", ["asn", "autonomous_system_number", "number"])
    reg_candidate_keys = p_cfg.get("registrar_candidate_keys", ["name", "handle", "organization"])

    extractor = tldextract.TLDExtract(suffix_list_urls=())
    asn_key_counter: Counter = Counter()
    reg_key_counter: Counter = Counter()

    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Parsing malware records...")
    malware_path = raw_dir / "malware.json"
    malware_recs, mal_total, mal_no_time, mal_malformed = stream_parse_file(
        malware_path,
        label=1,
        default_source="malware",
        extractor=extractor,
        normalize_to_e2ld=normalize_to_e2ld,
        asn_candidate_keys=asn_candidate_keys,
        registrar_candidate_keys=reg_candidate_keys,
        asn_key_counter=asn_key_counter,
        reg_key_counter=reg_key_counter,
        max_records=max_per_class
    )

    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Parsing Cisco Umbrella benign records...")
    umbrella_path = raw_dir / "benign_umbrella.json"
    umbrella_recs, umb_total, umb_no_time, umb_malformed = stream_parse_file(
        umbrella_path,
        label=0,
        default_source="Umbrella",
        extractor=extractor,
        normalize_to_e2ld=normalize_to_e2ld,
        asn_candidate_keys=asn_candidate_keys,
        registrar_candidate_keys=reg_candidate_keys,
        asn_key_counter=asn_key_counter,
        reg_key_counter=reg_key_counter,
        max_records=max_per_class
    )

    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Parsing CESNET benign records...")
    cesnet_path = raw_dir / "benign_cesnet.json"
    cesnet_recs, ces_total, ces_no_time, ces_malformed = stream_parse_file(
        cesnet_path,
        label=0,
        default_source="CESNET",
        extractor=extractor,
        normalize_to_e2ld=normalize_to_e2ld,
        asn_candidate_keys=asn_candidate_keys,
        registrar_candidate_keys=reg_candidate_keys,
        asn_key_counter=asn_key_counter,
        reg_key_counter=reg_key_counter,
        max_records=max_per_class
    )

    # Candidate ASN validation check
    total_asns_resolved = sum(asn_key_counter.values())
    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] ASN Keys resolved: {dict(asn_key_counter)} (Total: {total_asns_resolved})")
    if total_asns_resolved == 0:
        raise ValueError(
            f"Zero ASNs were resolved using candidate keys: {asn_candidate_keys}. "
            "Please check ASN structure in raw ip_data."
        )

    # Intra-class deduplication (by domain unit)
    def deduplicate_class(recs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for r in recs:
            d = r["domain"]
            if d not in seen:
                seen[d] = r
            else:
                # Keep earliest observation
                if r["t"] < seen[d]["t"]:
                    seen[d] = r
        return seen

    mal_kept = len(malware_recs)
    umb_kept = len(umbrella_recs)
    ces_kept = len(cesnet_recs)

    # Deduplicate within each class by retaining the earliest observation timestamp
    malware_dict = deduplicate_class(malware_recs)
    del malware_recs  # Free raw malware records list from memory

    benign_pool_dict = deduplicate_class(umbrella_recs + cesnet_recs)
    del umbrella_recs, cesnet_recs  # Free raw benign lists from memory
    gc.collect()  # Immediately reclaim host RAM

    # Cross-class deduplication: DROP any domain appearing in both classes
    cross_collisions = set(malware_dict.keys()).intersection(set(benign_pool_dict.keys()))
    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Dropping {len(cross_collisions)} cross-class domain collisions...")
    for d in cross_collisions:
        malware_dict.pop(d, None)
        benign_pool_dict.pop(d, None)

    clean_malware = list(malware_dict.values())
    del malware_dict
    clean_benign = list(benign_pool_dict.values())
    del benign_pool_dict
    gc.collect()

    # Monthly distributions
    malware_by_month: Dict[str, List[Dict[str, Any]]] = {}
    for r in clean_malware:
        malware_by_month.setdefault(r["t_month"], []).append(r)

    benign_by_month: Dict[str, List[Dict[str, Any]]] = {}
    for r in clean_benign:
        benign_by_month.setdefault(r["t_month"], []).append(r)

    overlapping_months = sorted(list(set(malware_by_month.keys()).intersection(set(benign_by_month.keys()))))
    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Overlapping months with both classes ({len(overlapping_months)}): {overlapping_months}")

    rng = np.random.default_rng(config.get("randomness", {}).get("seed", 42))

    # 1. Time-matched benign sampling over overlapping months
    time_matched_benign: List[Dict[str, Any]] = []
    time_matched_malware: List[Dict[str, Any]] = []
    for mo in overlapping_months:
        m_list = malware_by_month[mo]
        b_list = benign_by_month[mo]
        time_matched_malware.extend(m_list)
        if benign_ratio is None:
            time_matched_benign.extend(b_list)
        else:
            n_target = int(round(len(m_list) * benign_ratio))
            if len(b_list) <= n_target:
                time_matched_benign.extend(b_list)
            else:
                idx = rng.choice(len(b_list), size=n_target, replace=False)
                time_matched_benign.extend([b_list[i] for i in idx])

    # 2. Uniform benign sampling (for random / group splits across the entire benign pool)
    if benign_ratio is None:
        uniform_benign = clean_benign
    else:
        n_uniform_target = min(len(clean_benign), int(round(len(clean_malware) * benign_ratio)))
        if len(clean_benign) <= n_uniform_target:
            uniform_benign = clean_benign
        else:
            idx = rng.choice(len(clean_benign), size=n_uniform_target, replace=False)
            uniform_benign = [clean_benign[i] for i in idx]

    # Combine all unique sampled domains
    time_matched_domain_names = {r["domain"] for r in (time_matched_malware + time_matched_benign)}

    # Primary pool: malware + uniform benign
    primary_domains = clean_malware + uniform_benign
    for r in primary_domains:
        r["in_time_matched"] = (r["domain"] in time_matched_domain_names)

    # RFC-001 Dual-table dictionary-encoded Parquet export (domains_fqdn.parquet + domains_apex.parquet)
    df_fqdn, df_apex = export_dual_fqdn_apex_tables(primary_domains, output_dir)

    df_primary = pd.DataFrame(primary_domains)
    parquet_path = output_dir / "domains.parquet"
    df_primary.to_parquet(parquet_path, index=False)

    # Secondary pool: specifically time-matched cohort
    df_timematched = pd.DataFrame(time_matched_malware + time_matched_benign)
    tm_parquet_path = output_dir / "domains_timematched.parquet"
    df_timematched.to_parquet(tm_parquet_path, index=False)

    summary = {
        "is_fixture": is_fixture,
        "processed_at": datetime.now().isoformat(),
        "preserve_subdomains": preserve_subdomains,
        "normalize_to_e2LD": normalize_to_e2ld,
        "benign_ratio": benign_ratio,
        "raw_counts": {
            "malware": {"total": mal_total, "excluded_no_time": mal_no_time, "malformed": mal_malformed, "kept": mal_kept},
            "umbrella": {"total": umb_total, "excluded_no_time": umb_no_time, "malformed": mal_malformed, "kept": umb_kept},
            "cesnet": {"total": ces_total, "excluded_no_time": ces_no_time, "malformed": ces_malformed, "kept": ces_kept}
        },
        "asn_keys_used": dict(asn_key_counter),
        "registrar_keys_used": dict(reg_key_counter),
        "cross_class_dropped_count": len(cross_collisions),
        "overlapping_months": overlapping_months,
        "final_counts": {
            "malware": len(clean_malware),
            "benign_uniform": len(uniform_benign),
            "total_primary": len(df_primary),
            "fqdn_count": len(df_fqdn),
            "apex_count": len(df_apex),
            "time_matched_malware": len(time_matched_malware),
            "time_matched_benign": len(time_matched_benign),
            "total_timematched": len(df_timematched)
        }
    }

    summary_path = output_dir / "parsing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Parsing completed successfully.")
    print(f"  Primary Parquet:     {parquet_path} ({len(df_primary)} records)")
    print(f"  Time-matched Parquet: {tm_parquet_path} ({len(df_timematched)} records)")
    return summary
