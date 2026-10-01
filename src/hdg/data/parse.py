"""
Streaming parser and normalizer for Zenodo DomainRadar v2.
Extracts schema-faithful domain records, enforces leak-free normalization,
resolves candidate keys for ASN and registrar, performs cross-class deduplication,
handles dual certificate keys (CN and co-issuance), optional BGP prefix extraction,
and supports month-matched benign sampling.
"""

from collections import Counter
from datetime import datetime
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

logger = logging.getLogger(__name__)


def extract_leaf_certificate(tls_obj: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Finds the first non-root certificate (leaf) using is_root flag (not position).
    Returns (leaf_cert_key_cn, leaf_cert_key_coissue, validity_length_days).
    - leaf_cert_key_cn: hash(common_name, organization, country, validity_start, validity_end)
    - leaf_cert_key_coissue: hash(organization, country, validity_start, validity_end, valid_len) (NO common_name)
    Note: Zenodo certificates have no fingerprint/serial, so identical-metadata certs collide.
    """
    if not isinstance(tls_obj, dict):
        return None, None, None

    certs = tls_obj.get("certificates")
    if not isinstance(certs, list):
        return None, None, None

    leaf_cert = None
    for c in certs:
        if isinstance(c, dict) and not c.get("is_root", False):
            leaf_cert = c
            break

    if leaf_cert is None:
        return None, None, None

    cn = str(leaf_cert.get("common_name") or "")
    org = str(leaf_cert.get("organization") or "")
    country = str(leaf_cert.get("country") or "")
    v_start = parse_mongo_date(leaf_cert.get("validity_start"))
    v_end = parse_mongo_date(leaf_cert.get("validity_end"))
    v_start_str = v_start.isoformat() if v_start else ""
    v_end_str = v_end.isoformat() if v_end else ""
    valid_len = leaf_cert.get("valid_len")
    val_len_int = int(valid_len) if valid_len is not None else None

    # Key 1: with common_name (standard leaf cert identifier)
    meta_cn = f"{cn}|{org}|{country}|{v_start_str}|{v_end_str}"
    cert_hash_cn = hashlib.sha256(meta_cn.encode("utf-8")).hexdigest()[:16]

    # Key 2: co-issuance key (omits common_name to capture shared infrastructure / batch issuance)
    meta_coissue = f"{org}|{country}|{v_start_str}|{v_end_str}|{val_len_int}"
    cert_hash_coissue = hashlib.sha256(meta_coissue.encode("utf-8")).hexdigest()[:16]

    return cert_hash_cn, cert_hash_coissue, val_len_int


def resolve_asn(
    asn_obj: Optional[Dict[str, Any]],
    candidate_keys: List[str],
    asn_key_counter: Counter
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Resolves ASN number, organization, and optional BGP prefix/network from candidate keys.
    Logs which key was used.
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

    org = str(asn_obj.get("organization") or "").strip()
    prefix = str(asn_obj.get("network") or "").strip()
    return asn_num, (org if org else None), (prefix if prefix else None)


def resolve_registrar(
    rdap_obj: Optional[Dict[str, Any]],
    candidate_keys: List[str],
    registrar_key_counter: Counter
) -> Optional[str]:
    """
    Resolves registrar name from candidate entity keys inside rdap.entities.registrar[].
    """
    if not isinstance(rdap_obj, dict):
        return None

    entities = rdap_obj.get("entities")
    if not isinstance(entities, dict):
        return None

    reg_list = entities.get("registrar")
    if not isinstance(reg_list, list):
        return None

    for item in reg_list:
        if isinstance(item, dict):
            for k in candidate_keys:
                if k in item and item[k]:
                    val = str(item[k]).strip()
                    if val:
                        registrar_key_counter[k] += 1
                        return val
    return None


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

                # Source and family
                source = str(r.get("source") or default_source)
                # Malware family key conflict: read both r.get("malware_type") or r.get("malware")
                fam = r.get("malware_type") or r.get("malware")
                family = str(fam) if (label == 1 and fam is not None) else None

                # RDAP
                rdap = r.get("rdap")
                no_rdap = (rdap is None)
                t_reg = parse_mongo_date(rdap.get("registration_date")) if isinstance(rdap, dict) else None
                registrar = resolve_registrar(rdap, registrar_candidate_keys, reg_key_counter)

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

                # DNS Flags
                if isinstance(dns, dict):
                    a_recs = dns.get("A")
                    aaaa_recs = dns.get("AAAA")
                    has_a = bool(a_recs)
                    has_aaaa = bool(aaaa_recs)
                    no_ip = not (has_a or has_aaaa)
                    mx_dict = dns.get("MX")
                    mx_count = len(mx_dict) if isinstance(mx_dict, dict) else 0
                    remarks = dns.get("remarks") if isinstance(dns.get("remarks"), dict) else {}
                    has_spf = bool(remarks.get("has_spf", False))
                    has_dmarc = bool(remarks.get("has_dmarc", False))
                    has_dkim = bool(remarks.get("has_dkim", False))
                    dnssec = bool(dns.get("dnssec"))
                    ttls = dns.get("ttls") if isinstance(dns.get("ttls"), dict) else {}
                else:
                    no_ip = True
                    mx_count = 0
                    has_spf = False
                    has_dmarc = False
                    has_dkim = False
                    dnssec = False
                    ttls = {}

                # TLS & Certificates (both CN and co-issuance keys)
                tls = r.get("tls")
                no_tls = (tls is None)
                tls_proto = str(tls.get("protocol")) if (isinstance(tls, dict) and tls.get("protocol")) else None
                tls_cipher = str(tls.get("cipher")) if (isinstance(tls, dict) and tls.get("cipher")) else None
                certs = tls.get("certificates") if isinstance(tls, dict) else None
                cert_count = len(certs) if isinstance(certs, list) else 0
                leaf_cert_cn, leaf_cert_coissue, cert_valid_len = extract_leaf_certificate(tls)

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

                        asn_num, asn_org, bgp_prefix = resolve_asn(entry.get("asn"), asn_candidate_keys, asn_key_counter)
                        if asn_num is not None:
                            domain_asns.add(asn_num)

                        geo_obj = entry.get("geo")
                        country = str(geo_obj.get("country")) if isinstance(geo_obj, dict) and geo_obj.get("country") else None

                        ip_records.append({
                            "ip": ip_val,
                            "from_record": from_rec,
                            "asn": asn_num,
                            "asn_org": asn_org,
                            "country": country,
                            "prefix": bgp_prefix
                        })

                # Also check related_ips in dns.NS to retain NS-IP relations
                ns_ip_pairs: List[Tuple[str, str]] = []
                if isinstance(dns, dict) and isinstance(dns.get("NS"), dict):
                    for ns_host, ns_val in dns["NS"].items():
                        ns_clean = str(ns_host).strip().lower().rstrip(".")
                        if isinstance(ns_val, dict) and isinstance(ns_val.get("related_ips"), list):
                            for rel in ns_val["related_ips"]:
                                if isinstance(rel, dict) and rel.get("value"):
                                    ns_ip_pairs.append((ns_clean, str(rel["value"]).strip()))

                primary_asn = sorted(list(domain_asns))[0] if domain_asns else None

                rec = {
                    "domain": domain_unit,
                    "raw_domain": raw_domain,
                    "e2LD": e2ld,
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
                    "mx_count": mx_count,
                    "has_spf": has_spf,
                    "has_dmarc": has_dmarc,
                    "has_dkim": has_dkim,
                    "dnssec": dnssec,
                    "ttl_a": int(ttls.get("A", 0)) if isinstance(ttls, dict) else 0,
                    "ttl_ns": int(ttls.get("NS", 0)) if isinstance(ttls, dict) else 0,
                    "nameservers": sorted(list(ns_set)),
                    "registrar": registrar,
                    "tls_protocol": tls_proto,
                    "tls_cipher": tls_cipher,
                    "tls_cert_count": cert_count,
                    "leaf_cert_key": leaf_cert_cn,
                    "leaf_cert_key_cn": leaf_cert_cn,
                    "leaf_cert_key_coissue": leaf_cert_coissue,
                    "cert_valid_len": cert_valid_len,
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
    Writes data_processed/domains.parquet, data_processed/domains_timematched.parquet, and summary JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p_cfg = config.get("parsing", {})
    normalize_to_e2ld = p_cfg.get("normalize_to_e2LD", True)
    benign_ratio = float(p_cfg.get("benign_ratio", 2.0))
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

    malware_dict = deduplicate_class(malware_recs)
    benign_pool_dict = deduplicate_class(umbrella_recs + cesnet_recs)

    # Cross-class deduplication: DROP any domain appearing in both classes
    cross_collisions = set(malware_dict.keys()).intersection(set(benign_pool_dict.keys()))
    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Dropping {len(cross_collisions)} cross-class domain collisions...")
    for d in cross_collisions:
        malware_dict.pop(d, None)
        benign_pool_dict.pop(d, None)

    clean_malware = list(malware_dict.values())
    clean_benign = list(benign_pool_dict.values())

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
        n_target = int(round(len(m_list) * benign_ratio))
        if len(b_list) <= n_target:
            time_matched_benign.extend(b_list)
        else:
            idx = rng.choice(len(b_list), size=n_target, replace=False)
            time_matched_benign.extend([b_list[i] for i in idx])

    # 2. Uniform benign sampling (for random / group splits across the entire benign pool)
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
        "normalize_to_e2LD": normalize_to_e2ld,
        "benign_ratio": benign_ratio,
        "raw_counts": {
            "malware": {"total": mal_total, "excluded_no_time": mal_no_time, "malformed": mal_malformed, "kept": len(malware_recs)},
            "umbrella": {"total": umb_total, "excluded_no_time": umb_no_time, "malformed": mal_malformed, "kept": len(umbrella_recs)},
            "cesnet": {"total": ces_total, "excluded_no_time": ces_no_time, "malformed": ces_malformed, "kept": len(cesnet_recs)}
        },
        "asn_keys_used": dict(asn_key_counter),
        "registrar_keys_used": dict(reg_key_counter),
        "cross_class_dropped_count": len(cross_collisions),
        "overlapping_months": overlapping_months,
        "final_counts": {
            "malware": len(clean_malware),
            "benign_uniform": len(uniform_benign),
            "total_primary": len(df_primary),
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
