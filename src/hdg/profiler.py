"""
Dataset profiler for Zenodo DomainRadar v2.
Streams multi-GB JSON arrays with ijson, computing structural presence rates,
key counts, date distributions, and temporal feasibility.
"""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import ijson
import numpy as np
import tldextract


def parse_mongo_date(d: Any) -> Optional[datetime]:
    """
    Parses dates in mongoexport format:
    - {'$date': '2023-05-10T12:00:00Z'}
    - {'$date': {'$numberLong': '1683720000000'}}
    - '2023-05-10T12:00:00Z'
    - Unix timestamp int/float
    """
    if d is None:
        return None
    try:
        if isinstance(d, dict):
            if "$date" in d:
                val = d["$date"]
                if isinstance(val, dict) and "$numberLong" in val:
                    ms = int(val["$numberLong"])
                    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
                elif isinstance(val, (int, float)):
                    return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
                elif isinstance(val, str):
                    clean = val.rstrip("Z").split(".")[0]
                    return datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
        elif isinstance(d, (int, float)):
            return datetime.fromtimestamp(d / 1000.0, tz=timezone.utc)
        elif isinstance(d, str):
            clean = d.rstrip("Z").split(".")[0]
            return datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return None


class ClassProfileAccumulator:
    """Accumulates schema statistics across streaming records for a class/file."""

    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.top_keys: Counter = Counter()
        self.sources: Counter = Counter()
        self.malware_families: Counter = Counter()
        self.has_malware_key_count = 0
        self.has_malware_type_key_count = 0

        self.null_counts = {
            "dns": 0,
            "rdap": 0,
            "tls": 0,
            "ip_data": 0
        }

        self.sourced_on_months: Counter = Counter()
        self.evaluated_on_months: Counter = Counter()
        self.reg_date_months: Counter = Counter()

        self.sourced_on_dates: Set[str] = set()
        self.evaluated_on_dates: Set[str] = set()
        self.reg_dates: Set[str] = set()

        self.sourced_on_min: Optional[datetime] = None
        self.sourced_on_max: Optional[datetime] = None

        self.subdomain_count = 0
        self.e2ld_only_count = 0
        self.lengths: List[int] = []
        self.num_labels_list: List[int] = []
        self.tlds: Counter = Counter()
        self.no_a_aaaa_count = 0

        self.asn_keys: Counter = Counter()
        self.geo_keys: Counter = Counter()
        self.registrar_entity_keys: Counter = Counter()
        self.ns_structure_types: Counter = Counter()
        self.cert_field_presence: Counter = Counter()

        self.extractor = tldextract.TLDExtract(suffix_list_urls=())

    def update(self, r: Dict[str, Any]) -> None:
        self.count += 1

        # Top-level keys
        for k in r.keys():
            self.top_keys[k] += 1

        # Sources
        src = r.get("source")
        if src is not None:
            self.sources[str(src)] += 1

        # Malware family key conflict: malware vs malware_type
        has_m = "malware" in r
        has_mt = "malware_type" in r
        if has_m:
            self.has_malware_key_count += 1
        if has_mt:
            self.has_malware_type_key_count += 1

        fam = r.get("malware_type") or r.get("malware")
        if fam is not None:
            self.malware_families[str(fam)] += 1

        # Null tracking
        for k in ["dns", "rdap", "tls", "ip_data"]:
            if r.get(k) is None:
                self.null_counts[k] += 1

        # Date parsing
        t_src = parse_mongo_date(r.get("sourced_on"))
        if t_src:
            mo_str = t_src.strftime("%Y-%m")
            self.sourced_on_months[mo_str] += 1
            self.sourced_on_dates.add(t_src.strftime("%Y-%m-%d"))
            if self.sourced_on_min is None or t_src < self.sourced_on_min:
                self.sourced_on_min = t_src
            if self.sourced_on_max is None or t_src > self.sourced_on_max:
                self.sourced_on_max = t_src

        t_eval = parse_mongo_date(r.get("evaluated_on"))
        if t_eval:
            self.evaluated_on_months[t_eval.strftime("%Y-%m")] += 1
            self.evaluated_on_dates.add(t_eval.strftime("%Y-%m-%d"))

        rdap = r.get("rdap")
        if isinstance(rdap, dict):
            t_reg = parse_mongo_date(rdap.get("registration_date"))
            if t_reg:
                self.reg_date_months[t_reg.strftime("%Y-%m")] += 1
                self.reg_dates.add(t_reg.strftime("%Y-%m-%d"))

            # Registrar entity keys
            entities = rdap.get("entities")
            if isinstance(entities, dict):
                reg_list = entities.get("registrar")
                if isinstance(reg_list, list):
                    for item in reg_list:
                        if isinstance(item, dict):
                            for ek in item.keys():
                                self.registrar_entity_keys[ek] += 1

        # Domain lexical & TLD stats
        d_name = r.get("domain_name") or ""
        d_len = len(d_name)
        self.lengths.append(d_len)
        labels = d_name.split(".") if d_name else []
        self.num_labels_list.append(len(labels))

        ext_res = self.extractor(d_name)
        if ext_res.subdomain:
            self.subdomain_count += 1
        else:
            self.e2ld_only_count += 1

        tld = ext_res.suffix
        if tld:
            self.tlds[f".{tld}"] += 1

        # DNS analysis
        dns = r.get("dns")
        if isinstance(dns, dict):
            has_a = bool(dns.get("A"))
            has_aaaa = bool(dns.get("AAAA"))
            if not has_a and not has_aaaa:
                self.no_a_aaaa_count += 1

            ns_obj = dns.get("NS")
            if ns_obj is not None:
                self.ns_structure_types[type(ns_obj).__name__] += 1
        else:
            self.no_a_aaaa_count += 1

        # IP data analysis
        ip_data = r.get("ip_data")
        if isinstance(ip_data, list):
            for entry in ip_data:
                if isinstance(entry, dict):
                    asn = entry.get("asn")
                    if isinstance(asn, dict):
                        for ak in asn.keys():
                            self.asn_keys[ak] += 1
                    geo = entry.get("geo")
                    if isinstance(geo, dict):
                        for gk in geo.keys():
                            self.geo_keys[gk] += 1

        # TLS analysis
        tls = r.get("tls")
        if isinstance(tls, dict):
            certs = tls.get("certificates")
            if isinstance(certs, list):
                for cert in certs:
                    if isinstance(cert, dict):
                        for ck in cert.keys():
                            self.cert_field_presence[ck] += 1

    def summary(self) -> Dict[str, Any]:
        """Summarizes accumulated statistics into a serializable dict."""
        cnt = max(self.count, 1)
        len_arr = np.array(self.lengths) if self.lengths else np.array([0])
        labels_arr = np.array(self.num_labels_list) if self.num_labels_list else np.array([0])

        return {
            "record_count": self.count,
            "top_level_key_presence_rate": {k: round(v / cnt, 4) for k, v in self.top_keys.most_common()},
            "malware_key_count": self.has_malware_key_count,
            "malware_type_key_count": self.has_malware_type_key_count,
            "null_rates": {k: round(v / cnt, 4) for k, v in self.null_counts.items()},
            "top_sources": dict(self.sources.most_common(30)),
            "top_malware_families": dict(self.malware_families.most_common(50)),
            "sourced_on_months": dict(sorted(self.sourced_on_months.items())),
            "evaluated_on_months": dict(sorted(self.evaluated_on_months.items())),
            "registration_date_months": dict(sorted(self.reg_date_months.items())),
            "distinct_sourced_on_dates_count": len(self.sourced_on_dates),
            "distinct_evaluated_on_dates_count": len(self.evaluated_on_dates),
            "distinct_reg_dates_count": len(self.reg_dates),
            "sourced_on_min": self.sourced_on_min.isoformat() if self.sourced_on_min else None,
            "sourced_on_max": self.sourced_on_max.isoformat() if self.sourced_on_max else None,
            "e2ld_only_share": round(self.e2ld_only_count / cnt, 4),
            "subdomain_share": round(self.subdomain_count / cnt, 4),
            "length_stats": {
                "mean": round(float(np.mean(len_arr)), 2),
                "std": round(float(np.std(len_arr)), 2),
                "median": float(np.median(len_arr)),
                "p25": float(np.percentile(len_arr, 25)),
                "p75": float(np.percentile(len_arr, 75))
            },
            "num_labels_stats": {
                "mean": round(float(np.mean(labels_arr)), 2),
                "std": round(float(np.std(labels_arr)), 2),
                "median": float(np.median(labels_arr))
            },
            "top_tlds": dict(self.tlds.most_common(20)),
            "no_a_or_aaaa_share": round(self.no_a_aaaa_count / cnt, 4),
            "ip_asn_observed_keys": dict(self.asn_keys),
            "ip_geo_observed_keys": dict(self.geo_keys),
            "rdap_registrar_entity_observed_keys": dict(self.registrar_entity_keys),
            "dns_ns_structure_types": dict(self.ns_structure_types),
            "tls_cert_field_presence": dict(self.cert_field_presence)
        }


def profile_file(path: Path, acc: ClassProfileAccumulator) -> None:
    """Streams a JSON array file with ijson, passing records to the accumulator."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found for profiling: {path}")

    total_seen = 0
    malformed_count = 0
    with open(path, "rb") as f:
        # Avoid Decimal conversion with use_float=True
        records = ijson.items(f, "item", use_float=True)
        for rec in records:
            total_seen += 1
            if not isinstance(rec, dict):
                malformed_count += 1
                continue
            try:
                acc.update(rec)
            except Exception as e:
                malformed_count += 1
                if total_seen > 100 and (malformed_count / total_seen) > 0.01:
                    raise ValueError(
                        f"Malformed records exceeded 1% threshold in {path.name}: "
                        f"{malformed_count}/{total_seen} malformed records. Last error: {e}"
                    )
                continue



def profile_dataset(
    raw_dir: Path,
    output_dir: Path,
    min_distinct_months: int = 4,
    is_fixture: bool = False
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Profiles all dataset files and generates:
    1. profile_report.json
    2. profile_report.md (<= 200 lines)
    3. time_split_valid.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    malware_path = raw_dir / "malware.json"
    umbrella_path = raw_dir / "benign_umbrella.json"
    cesnet_path = raw_dir / "benign_cesnet.json"
    schema_path = raw_dir / "data_schema.json"

    malware_acc = ClassProfileAccumulator("malware")
    umbrella_acc = ClassProfileAccumulator("benign_umbrella")
    cesnet_acc = ClassProfileAccumulator("benign_cesnet")

    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Profiling {malware_path.name}...")
    profile_file(malware_path, malware_acc)
    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Profiling {umbrella_path.name}...")
    profile_file(umbrella_path, umbrella_acc)
    print(f"[{'FIXTURE' if is_fixture else 'REAL'}] Profiling {cesnet_path.name}...")
    profile_file(cesnet_path, cesnet_acc)

    # Read data_schema.json if available
    rdap_entity_def: Optional[Dict[str, Any]] = None
    if schema_path.exists():
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                s_json = json.load(f)
                rdap_entity_def = s_json.get("$defs", {}).get("rdapEntity")
        except Exception as e:
            rdap_entity_def = {"error": str(e)}

    malware_summary = malware_acc.summary()
    umbrella_summary = umbrella_acc.summary()
    cesnet_summary = cesnet_acc.summary()

    # Time split validity check
    malware_months = set(malware_summary["sourced_on_months"].keys())
    benign_months = set(umbrella_summary["sourced_on_months"].keys()).union(set(cesnet_summary["sourced_on_months"].keys()))
    overlap_months = malware_months.intersection(benign_months)

    time_split_valid = True
    reasons: List[str] = []

    if len(malware_months) < min_distinct_months:
        time_split_valid = False
        reasons.append(f"Malware has {len(malware_months)} distinct sourced_on months (< {min_distinct_months}).")

    if len(benign_months) < min_distinct_months:
        time_split_valid = False
        reasons.append(f"Benign has {len(benign_months)} distinct sourced_on months (< {min_distinct_months}).")

    if len(overlap_months) == 0:
        time_split_valid = False
        reasons.append("Zero temporal overlap in sourced_on months between malware and benign.")

    time_split_data = {
        "time_split_valid": time_split_valid,
        "min_required_months": min_distinct_months,
        "malware_months_count": len(malware_months),
        "benign_months_count": len(benign_months),
        "overlapping_months_count": len(overlap_months),
        "overlapping_months": sorted(list(overlap_months)),
        "reasons": reasons
    }

    full_report = {
        "metadata": {
            "is_fixture": is_fixture,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prefix": "FIXTURE - meaningless" if is_fixture else "REAL"
        },
        "time_split_valid": time_split_data,
        "data_schema_rdap_entity": rdap_entity_def,
        "classes": {
            "malware": malware_summary,
            "benign_umbrella": umbrella_summary,
            "benign_cesnet": cesnet_summary
        }
    }

    # Write profile_report.json
    json_path = output_dir / "profile_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Write time_split_valid.json
    time_split_path = output_dir / "time_split_valid.json"
    with open(time_split_path, "w", encoding="utf-8") as f:
        json.dump(time_split_data, f, indent=2)

    # Build COMPACT Markdown (<= 200 lines)
    md_lines: List[str] = []
    pfx = "[FIXTURE - meaningless] " if is_fixture else ""
    md_lines.append(f"# {pfx}Dataset Profile Report: Zenodo DomainRadar v2")
    md_lines.append(f"*Generated at: {full_report['metadata']['generated_at']}*")
    md_lines.append("")
    md_lines.append("## 1. Summary Counts & Schema Keys")
    md_lines.append("| Dataset File | Record Count | DNS Null % | RDAP Null % | TLS Null % | IP Null % | No A/AAAA % |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for name, s in [("malware.json", malware_summary), ("benign_umbrella.json", umbrella_summary), ("benign_cesnet.json", cesnet_summary)]:
        nr = s["null_rates"]
        md_lines.append(
            f"| {name} | {s['record_count']:,} | {nr['dns']*100:.1f}% | {nr['rdap']*100:.1f}% | {nr['tls']*100:.1f}% | {nr['ip_data']*100:.1f}% | {s['no_a_or_aaaa_share']*100:.1f}% |"
        )
    md_lines.append("")
    md_lines.append("### Malware Family Key Conflict Inspection")
    md_lines.append(f"- `malware` key present in: {malware_summary['malware_key_count']} records")
    md_lines.append(f"- `malware_type` key present in: {malware_summary['malware_type_key_count']} records")
    md_lines.append("")
    md_lines.append("## 2. Temporal Feasibility & Split Validity")
    md_lines.append(f"- **Time Split Valid**: `{time_split_valid}`")
    if reasons:
        for r_reason in reasons:
            md_lines.append(f"  - *Reason*: {r_reason}")
    md_lines.append(f"- Distinct Months: Malware={len(malware_months)}, Benign={len(benign_months)}, Overlapping={len(overlap_months)}")
    md_lines.append(f"- Sourced_on Range (Malware): {malware_summary['sourced_on_min']} to {malware_summary['sourced_on_max']}")
    md_lines.append("")
    md_lines.append("## 3. Domain Lexical & Subdomain Distributions")
    md_lines.append("| Class | Subdomain % | e2LD % | Length (mean ± std) | Labels (mean ± std) |")
    md_lines.append("|---|---|---|---|---|")
    for name, s in [("Malware", malware_summary), ("Umbrella", umbrella_summary), ("CESNET", cesnet_summary)]:
        l_stat = s["length_stats"]
        lb_stat = s["num_labels_stats"]
        md_lines.append(
            f"| {name} | {s['subdomain_share']*100:.1f}% | {s['e2ld_only_share']*100:.1f}% | {l_stat['mean']} ± {l_stat['std']} | {lb_stat['mean']} ± {lb_stat['std']} |"
        )
    md_lines.append("")
    md_lines.append("## 4. Key Counters & Schema Verification")
    md_lines.append(f"- **ASN Candidate Keys Found**: {dict(Counter(malware_summary['ip_asn_observed_keys']) + Counter(umbrella_summary['ip_asn_observed_keys']))}")
    md_lines.append(f"- **Registrar Entity Keys Found**: {dict(Counter(malware_summary['rdap_registrar_entity_observed_keys']) + Counter(umbrella_summary['rdap_registrar_entity_observed_keys']))}")
    md_lines.append(f"- **DNS.NS Structure Types**: {dict(Counter(malware_summary['dns_ns_structure_types']) + Counter(umbrella_summary['dns_ns_structure_types']))}")
    md_lines.append(f"- **TLS Cert Fields**: {list(malware_summary['tls_cert_field_presence'].keys())}")
    md_lines.append("")
    md_lines.append("## 5. Top Sources & Top Malware Families")
    top_src_str = ", ".join([f"{k} ({v})" for k, v in list(malware_summary["top_sources"].items())[:8]])
    top_fam_str = ", ".join([f"{k} ({v})" for k, v in list(malware_summary["top_malware_families"].items())[:8]])
    md_lines.append(f"- **Malware Sources**: {top_src_str}")
    md_lines.append(f"- **Malware Families**: {top_fam_str}")
    md_lines.append("")
    md_lines.append("## 6. Runtime $defs.rdapEntity Schema")
    md_lines.append("```json")
    md_lines.append(json.dumps(rdap_entity_def, indent=2))
    md_lines.append("```")

    # Ensure strictly <= 200 lines
    if len(md_lines) > 200:
        md_lines = md_lines[:199] + ["... [truncated for compact chat sharing]"]

    md_content = "\n".join(md_lines) + "\n"
    md_path = output_dir / "profile_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return full_report, md_content, time_split_data
