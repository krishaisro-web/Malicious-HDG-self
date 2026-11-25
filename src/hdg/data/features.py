"""
Malicious-HDG: Named Feature Schema Registry & Zero-Leakage Multi-Modal Feature Extractors.

Implements RFC-003 (Gates 1-2):
1. Named `DOMAIN_FEATURE_SCHEMA` (30 features, or 31 with `has_subdomain`) eliminating all
   hardcoded column index magic numbers across graph construction, ablations, and attacks.
2. TLD-stripped domain body extraction + static frozen English bigram log-probability prior
   + compact uint8 ASCII byte-buffer encoding (`[N, 64]`) for end-to-end Char-CNN.
3. Multi-dimensional intrinsic node feature extractors for `ip`, `nameserver`, `asn`,
   `registrar`, `certificate`, and `prefix` using deterministic, label-free predicates
   and bounded feature hashing (zero test-time target encoding or label leakage).
"""

from collections import Counter
import hashlib
import ipaddress
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tldextract
import torch


# ---------------------------------------------------------------------------
# 1. Named Feature Schema Registry
# ---------------------------------------------------------------------------

# Base 22 non-lexical domain features (exact RFC-003 Section 2.2 order)
DOMAIN_NON_LEXICAL_SCHEMA: List[Tuple[str, Tuple[str, ...]]] = [
    # DNS Resolution & Cardinality (10)
    ("no_ip", ("non_lexical", "missingness")),
    ("a_count_log", ("non_lexical",)),
    ("aaaa_count_log", ("non_lexical",)),
    ("ip_diversity_ratio", ("non_lexical",)),
    ("ns_count_log", ("non_lexical", "controllable")),
    ("mx_count", ("non_lexical", "controllable")),
    ("has_spf", ("non_lexical", "controllable")),
    ("has_dmarc", ("non_lexical", "controllable")),
    ("has_dkim", ("non_lexical", "controllable")),
    ("dnssec", ("non_lexical", "controllable")),
    # DNS TTL Dynamics (3)
    ("ttl_a_log", ("non_lexical", "controllable")),
    ("ttl_ns_log", ("non_lexical", "controllable")),
    ("ttl_ratio", ("non_lexical", "controllable")),
    # RDAP Lifecycle & Age (4)
    ("no_rdap", ("non_lexical", "missingness")),
    ("domain_age_days_log", ("non_lexical",)),
    ("is_fresh_lt_30d", ("non_lexical",)),
    ("is_established_gt_365d", ("non_lexical",)),
    # TLS Security Posture (5)
    ("no_tls", ("non_lexical", "missingness")),
    ("tls_cert_count", ("non_lexical", "controllable")),
    ("cert_valid_len_log", ("non_lexical", "controllable")),
    ("tls_modern_proto", ("non_lexical", "controllable")),
    ("cipher_aead_flag", ("non_lexical", "controllable")),
]

# 8 TLD-stripped lexical & DGA features
DOMAIN_LEXICAL_SCHEMA: List[Tuple[str, Tuple[str, ...]]] = [
    ("body_len", ("lexical",)),
    ("body_entropy", ("lexical",)),
    ("digit_ratio", ("lexical",)),
    ("vowel_ratio", ("lexical",)),
    ("longest_consonant_run", ("lexical",)),
    ("hyphen_ratio", ("lexical",)),
    ("bigram_en_logprob", ("lexical",)),
    ("punycode_or_brand_token", ("lexical",)),
]

# Infrastructure node feature schemas (first 2 cols are split-computed [log_deg, iif_weight])
INFRA_FEATURE_SCHEMAS: Dict[str, List[str]] = {
    "ip": [
        "log_deg", "iif_weight",
        "is_ipv6", "is_private_bogon", "prefix_len_norm", "geo_missing",
        "lat_norm", "lon_norm",
        "country_hash_0", "country_hash_1", "country_hash_2", "country_hash_3",
    ],
    "nameserver": [
        "log_deg", "iif_weight",
        "ns_len", "ns_entropy", "is_major_public_dns_or_cdn",
        "ns_num_glue_ips_log", "ns_num_distinct_asns_log",
        "ns_geo_diversity", "ns_org_is_cloud", "tld_is_cc",
    ],
    "asn": [
        "log_deg", "iif_weight",
        "asn_number_log", "as_org_missing",
        "kw_cloud_cdn", "kw_hosting_vps", "kw_telecom_isp", "kw_edu_gov",
        "observed_prefix_count_log", "country_risk_agnostic_hash",
    ],
    "registrar": [
        "log_deg", "iif_weight",
        "has_whois_server", "has_abuse_email", "has_url", "has_tel",
        "name_entropy", "is_icann_privacy_or_major",
    ],
    "certificate": [
        "log_deg", "iif_weight",
        "valid_len_log", "is_short_90d", "is_long_gt_365d", "extension_count",
        "ca_is_free_auto", "ca_is_paid_ev", "issuer_org_missing", "is_self_signed_pattern",
    ],
    "prefix": [
        "log_deg", "iif_weight",
        "is_ipv6_prefix", "prefix_len_norm",
    ],
    "apex_domain": [
        "log_deg", "iif_weight",
        "subdomain_fanout_log", "is_public_suffix",
        "no_rdap", "reg_age_days_log", "is_fresh_lt_30d", "is_established_gt_365d",
        "ns_count_log", "has_whois_server", "has_abuse_email", "whois_stability_score",
        "tld_is_com_net_org", "tld_is_cc", "tld_is_new_gtld", "apex_len",
    ],
}

INFRA_TOTAL_DIMS: Dict[str, int] = {nt: len(cols) for nt, cols in INFRA_FEATURE_SCHEMAS.items()}
INFRA_INTRINSIC_DIMS: Dict[str, int] = {nt: max(0, len(cols) - 2) for nt, cols in INFRA_FEATURE_SCHEMAS.items()}


def get_domain_feature_schema(
    use_lexical: bool = True,
    include_has_subdomain: bool = False,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """Returns ordered `(name, groups)` tuples for domain features."""
    schema = list(DOMAIN_NON_LEXICAL_SCHEMA)
    if include_has_subdomain:
        schema.append(("has_subdomain", ("non_lexical",)))
    if use_lexical:
        schema.extend(DOMAIN_LEXICAL_SCHEMA)
    return schema


def get_domain_feature_names(
    use_lexical: bool = True,
    include_has_subdomain: bool = False,
) -> List[str]:
    """Returns ordered domain feature names."""
    return [name for name, _ in get_domain_feature_schema(use_lexical=use_lexical, include_has_subdomain=include_has_subdomain)]


def get_feature_indices(
    node_type: str,
    group: str,
    include_has_subdomain: bool = False,
    dim: Optional[int] = None,
) -> List[int]:
    """
    Dynamically resolves column indices for a named feature group.
    Supports both the RFC-003 30/31-dim domain/fqdn schema and legacy 12/13/17/18-dim test tensors.
    """
    if node_type not in ("domain", "fqdn"):
        cols = INFRA_FEATURE_SCHEMAS.get(node_type, [])
        if group == "all":
            return list(range(len(cols)))
        if group == "split":
            return [0, 1] if len(cols) >= 2 else [0]
        if group == "intrinsic":
            return list(range(2, len(cols)))
        return []

    # Legacy compatibility for unit tests using minimal 12/13/17/18-dim domain tensors
    if dim is not None and dim in (12, 13, 17, 18):
        non_lex_dim = 13 if dim in (13, 18) else 12
        if group == "non_lexical":
            return list(range(non_lex_dim))
        if group == "lexical":
            return list(range(non_lex_dim, dim))
        if group == "controllable":
            return [3, 4, 5, 6, 8, 9, 10, 11]
        if group == "missingness":
            return [0, 1, 2]
        if group == "all":
            return list(range(dim))

    if dim == 23 or dim == 31:
        include_has_subdomain = True

    schema = get_domain_feature_schema(use_lexical=True, include_has_subdomain=include_has_subdomain)
    if group == "all":
        indices = list(range(len(schema)))
    else:
        indices = [i for i, (_, groups) in enumerate(schema) if group in groups]
    if dim is not None:
        indices = [i for i in indices if i < dim]
    return indices


# ---------------------------------------------------------------------------
# 2. Static Linguistic Priors & TLD-Stripped Body Helpers
# ---------------------------------------------------------------------------

# Static English character bigram probability table (top common letter transitions, normalized frequencies)
# Zero dataset label/source dependence — external linguistic prior only.
_COMMON_EN_BIGRAMS: Dict[str, float] = {
    "th": 0.0356, "he": 0.0307, "in": 0.0243, "er": 0.0205, "an": 0.0199,
    "re": 0.0185, "on": 0.0176, "at": 0.0149, "en": 0.0145, "nd": 0.0135,
    "ti": 0.0134, "es": 0.0134, "or": 0.0128, "te": 0.0120, "of": 0.0117,
    "ed": 0.0117, "is": 0.0113, "it": 0.0112, "al": 0.0109, "ar": 0.0107,
    "st": 0.0105, "to": 0.0104, "nt": 0.0104, "ng": 0.0095, "se": 0.0093,
    "ha": 0.0093, "as": 0.0087, "ou": 0.0087, "io": 0.0083, "le": 0.0083,
    "ve": 0.0083, "co": 0.0079, "me": 0.0079, "de": 0.0076, "hi": 0.0076,
    "ri": 0.0073, "ro": 0.0073, "ic": 0.0070, "ne": 0.0069, "ea": 0.0069,
    "ra": 0.0069, "ce": 0.0065, "li": 0.0062, "ch": 0.0060, "ll": 0.0058,
    "be": 0.0058, "ma": 0.0057, "si": 0.0055, "om": 0.0055, "ur": 0.0054,
    "ca": 0.0054, "el": 0.0053, "ta": 0.0053, "la": 0.0053, "ns": 0.0051,
    "di": 0.0050, "fo": 0.0049, "ho": 0.0049, "pe": 0.0048, "ec": 0.0048,
    "pr": 0.0047, "no": 0.0047, "ct": 0.0046, "us": 0.0045, "ac": 0.0045,
    "ot": 0.0044, "il": 0.0043, "tr": 0.0043, "ly": 0.0042, "nc": 0.0042,
    "et": 0.0041, "ut": 0.0041, "ss": 0.0041, "so": 0.0040, "rs": 0.0040,
    "un": 0.0039, "lo": 0.0039, "wa": 0.0039, "ge": 0.0038, "ie": 0.0038,
    "wh": 0.0038, "ee": 0.0038, "wi": 0.0037, "em": 0.0037, "ad": 0.0037,
    "ol": 0.0036, "rt": 0.0036, "po": 0.0036, "we": 0.0036, "na": 0.0035,
    "ul": 0.0035, "ni": 0.0034, "ts": 0.0034, "mo": 0.0034, "ow": 0.0033,
    "pa": 0.0032, "im": 0.0032, "mi": 0.0032, "ai": 0.0032, "sh": 0.0031,
    "ir": 0.0031, "su": 0.0031, "id": 0.0030, "os": 0.0030, "iv": 0.0029,
    "ia": 0.0029, "am": 0.0029, "fi": 0.0029, "ci": 0.0028, "vi": 0.0028,
    "pl": 0.0027, "ig": 0.0027, "tu": 0.0027, "ev": 0.0027, "ld": 0.0027,
    "ry": 0.0026, "mp": 0.0026, "fe": 0.0025, "bl": 0.0024, "ab": 0.0024,
    "gh": 0.0024, "ty": 0.0024, "op": 0.0023, "wo": 0.0023, "sa": 0.0023,
    "ay": 0.0023, "ex": 0.0022, "ke": 0.0022, "fr": 0.0022, "oo": 0.0022,
    "av": 0.0022, "ag": 0.0021, "if": 0.0021, "ap": 0.0021, "gr": 0.0020,
    "od": 0.0020, "bo": 0.0020, "sp": 0.0020, "rd": 0.0019, "do": 0.0019,
    "uc": 0.0019, "bu": 0.0019, "ei": 0.0019, "ov": 0.0018, "by": 0.0018,
    "rm": 0.0018, "ep": 0.0018, "tt": 0.0018, "oc": 0.0017, "fa": 0.0017,
    "ef": 0.0017, "cu": 0.0017, "rn": 0.0016, "sc": 0.0016, "gi": 0.0016,
    "da": 0.0016, "yo": 0.0016, "cr": 0.0015, "cl": 0.0015, "du": 0.0015,
    "ga": 0.0015, "qu": 0.0015, "ue": 0.0015, "ff": 0.0015, "ba": 0.0014,
    "up": 0.0014, "au": 0.0014, "va": 0.0014, "go": 0.0014, "pp": 0.0014,
    "mb": 0.0013, "br": 0.0013, "ck": 0.0013, "my": 0.0012, "dr": 0.0012,
    "ok": 0.0012, "bi": 0.0012, "gl": 0.0011, "fl": 0.0011, "rk": 0.0011,
}
_EN_BIGRAM_LOGP: Dict[str, float] = {k: math.log(v) for k, v in _COMMON_EN_BIGRAMS.items()}
_UNK_BIGRAM_LOGP: float = math.log(1e-4)

_BRAND_LURE_TOKENS = (
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "apple", "microsoft", "office365", "amazon", "google",
    "wallet", "support", "helpdesk", "auth", "webmail", "portal", "billing",
)

_OFFLINE_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def extract_domain_body(domain_str: str) -> str:
    """Strips public suffix / TLD from a domain name so lexical stats cannot memorize feed TLDs."""
    s = (domain_str or "").strip().lower().rstrip(".")
    if not s:
        return ""
    ext = _OFFLINE_TLD_EXTRACTOR(s)
    if ext.subdomain and ext.domain:
        return f"{ext.subdomain}.{ext.domain}"
    if ext.domain:
        return ext.domain
    if "." in s:
        return s.rsplit(".", 1)[0]
    return s


def compute_shannon_entropy(s: str) -> float:
    """Computes Shannon character entropy in bits."""
    if not s:
        return 0.0
    cnt = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())


def compute_longest_consonant_run(s: str) -> int:
    """Computes maximum consecutive consonant run length."""
    vowels_and_special = set("aeiou0123456789-._")
    max_run = 0
    cur_run = 0
    for ch in s.lower():
        if ch.isalpha() and ch not in vowels_and_special:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run


def compute_english_bigram_logprob(body: str) -> float:
    """Computes mean English character bigram log-probability over domain body."""
    cleaned = "".join(ch for ch in body.lower() if ch.isalpha())
    if len(cleaned) < 2:
        return _UNK_BIGRAM_LOGP
    total = 0.0
    n_pairs = len(cleaned) - 1
    for i in range(n_pairs):
        total += _EN_BIGRAM_LOGP.get(cleaned[i : i + 2], _UNK_BIGRAM_LOGP)
    return float(total / n_pairs)


def compute_punycode_or_brand_token(body: str) -> float:
    """
    Returns 1.0 if domain body contains Punycode (`xn--`), phishing/brand lure tokens,
    or >= 3 alpha-digit transitions (obfuscated DGA mixing); else 0.0.
    """
    b = body.lower()
    if "xn--" in b:
        return 1.0
    if any(tok in b for tok in _BRAND_LURE_TOKENS):
        return 1.0
    transitions = 0
    for i in range(len(b) - 1):
        c1, c2 = b[i], b[i + 1]
        if (c1.isalpha() and c2.isdigit()) or (c1.isdigit() and c2.isalpha()):
            transitions += 1
            if transitions >= 3:
                return 1.0
    return 0.0


def encode_domain_char_bytes(bodies: Sequence[str], max_len: int = 64) -> torch.Tensor:
    """
    Encodes TLD-stripped domain bodies into a compact `uint8` tensor of shape `[N, max_len]`.
    Memory footprint at 931k domains: 931,000 * 64 * 1B = 59.58 MB.
    """
    n = len(bodies)
    buf = np.zeros((n, max_len), dtype=np.uint8)
    for i, s in enumerate(bodies):
        raw = (s or "").encode("ascii", errors="replace")[:max_len]
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8)
            arr = np.clip(arr, 1, 127)
            buf[i, : len(arr)] = arr
    return torch.from_numpy(buf)


# ---------------------------------------------------------------------------
# 3. Domain Feature Extraction (Vectorized + Single-Row)
# ---------------------------------------------------------------------------

def _parse_iso_days_diff(t_eff_list: List[Any], t_reg_list: List[Any]) -> np.ndarray:
    """Vectorized age-in-days calculation max(0, (t_eff - t_reg).days) with zero leakage."""
    n = len(t_eff_list)
    ages = np.zeros(n, dtype=np.float32)
    has_age = np.zeros(n, dtype=bool)
    eff_ts = pd.to_datetime(pd.Series(t_eff_list), errors="coerce", utc=True)
    reg_ts = pd.to_datetime(pd.Series(t_reg_list), errors="coerce", utc=True)
    valid = eff_ts.notna() & reg_ts.notna()
    if valid.any():
        delta_days = (eff_ts[valid] - reg_ts[valid]).dt.total_seconds().to_numpy() / 86400.0
        delta_days = np.maximum(0.0, delta_days).astype(np.float32)
        ages[valid.to_numpy()] = delta_days
        has_age[valid.to_numpy()] = True
    return ages, has_age


def extract_domain_features_vectorized(
    df: pd.DataFrame,
    use_lexical: bool = True,
    include_has_subdomain: bool = False,
) -> np.ndarray:
    """
    Extracts the 30-dim (or 31-dim with `has_subdomain`) RFC-003 domain feature matrix.
    All lexical features are computed strictly on the TLD-stripped domain body.
    """
    n = len(df)
    if n == 0:
        d = len(get_domain_feature_schema(use_lexical=use_lexical, include_has_subdomain=include_has_subdomain))
        return np.empty((0, d), dtype=np.float32)

    def _col_series(col_name: str, default_val: Any) -> pd.Series:
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series([default_val] * n, index=df.index)

    # Resolution & cardinality
    no_ip_arr = _col_series("no_ip", False).astype(np.float32).to_numpy()

    if "a_count" in df.columns:
        a_cnt = _col_series("a_count", 0).to_numpy(dtype=np.float32)
    elif "resolved_ips" in df.columns:
        a_cnt = np.array(
            [len(x) if isinstance(x, (list, np.ndarray)) else 0 for x in df["resolved_ips"].tolist()],
            dtype=np.float32,
        )
    else:
        a_cnt = np.zeros(n, dtype=np.float32)

    if "aaaa_count" in df.columns:
        aaaa_cnt = _col_series("aaaa_count", 0).to_numpy(dtype=np.float32)
    else:
        aaaa_cnt = np.zeros(n, dtype=np.float32)

    a_count_log = np.log1p(np.maximum(0.0, a_cnt)).astype(np.float32)
    aaaa_count_log = np.log1p(np.maximum(0.0, aaaa_cnt)).astype(np.float32)

    if "asns" in df.columns and "resolved_ips" in df.columns:
        asns_list = df["asns"].tolist()
        ips_list = df["resolved_ips"].tolist()
        ip_div = np.array(
            [
                float(len(a_l) / max(1, len(i_l)))
                if isinstance(a_l, (list, np.ndarray)) and isinstance(i_l, (list, np.ndarray)) and len(i_l) > 0
                else 0.0
                for a_l, i_l in zip(asns_list, ips_list)
            ],
            dtype=np.float32,
        )
    else:
        ip_div = (1.0 - no_ip_arr).astype(np.float32)

    if "nameservers" in df.columns:
        ns_cnt = np.array(
            [len(x) if isinstance(x, (list, np.ndarray)) else 0 for x in df["nameservers"].tolist()],
            dtype=np.float32,
        )
    else:
        ns_cnt = np.zeros(n, dtype=np.float32)
    ns_count_log = np.log1p(np.maximum(0.0, ns_cnt)).astype(np.float32)

    mx_count_arr = _col_series("mx_count", 0).astype(np.float32).to_numpy()
    has_spf_arr = _col_series("has_spf", False).astype(np.float32).to_numpy()
    has_dmarc_arr = _col_series("has_dmarc", False).astype(np.float32).to_numpy()
    has_dkim_arr = _col_series("has_dkim", False).astype(np.float32).to_numpy()
    dnssec_arr = _col_series("dnssec", False).astype(np.float32).to_numpy()

    # TTL Dynamics
    ttl_a_raw = np.maximum(0.0, _col_series("ttl_a", 0).to_numpy(dtype=np.float32))
    ttl_ns_raw = np.maximum(0.0, _col_series("ttl_ns", 0).to_numpy(dtype=np.float32))
    ttl_a_log = np.log1p(ttl_a_raw).astype(np.float32)
    ttl_ns_log = np.log1p(ttl_ns_raw).astype(np.float32)
    ttl_ratio = np.log1p(ttl_a_raw / np.maximum(ttl_ns_raw, 1.0)).astype(np.float32)

    # RDAP Lifecycle & Age
    no_rdap_arr = _col_series("no_rdap", False).astype(np.float32).to_numpy()
    if "t_reg" in df.columns and ("t_eval" in df.columns or "t" in df.columns):
        t_eff_col = df["t_eval"].fillna(df["t"]) if ("t_eval" in df.columns and "t" in df.columns) else (
            df["t_eval"] if "t_eval" in df.columns else df["t"]
        )
        ages, has_age = _parse_iso_days_diff(t_eff_col.tolist(), df["t_reg"].tolist())
        domain_age_days_log = np.log1p(ages).astype(np.float32)
        is_fresh_lt_30d = np.where(has_age & (ages < 30.0), 1.0, 0.0).astype(np.float32)
        is_established_gt_365d = np.where(has_age & (ages > 365.0), 1.0, 0.0).astype(np.float32)
    else:
        domain_age_days_log = np.zeros(n, dtype=np.float32)
        is_fresh_lt_30d = np.zeros(n, dtype=np.float32)
        is_established_gt_365d = np.zeros(n, dtype=np.float32)

    # TLS Security Posture
    no_tls_arr = _col_series("no_tls", False).astype(np.float32).to_numpy()
    tls_cert_count_arr = _col_series("tls_cert_count", 0).astype(np.float32).to_numpy()
    cert_valid_len_log = np.log1p(np.maximum(0.0, _col_series("cert_valid_len", 0).to_numpy(dtype=np.float32)))

    if "tls_protocol" in df.columns:
        protos = [str(p or "").lower() for p in df["tls_protocol"].tolist()]
        tls_modern_proto = np.array([1.0 if ("1.3" in p or "1.2" in p) else 0.0 for p in protos], dtype=np.float32)
    else:
        tls_modern_proto = np.zeros(n, dtype=np.float32)

    if "tls_cipher" in df.columns:
        ciphers = [str(c or "").upper() for c in df["tls_cipher"].tolist()]
        cipher_aead_flag = np.array(
            [1.0 if any(tok in c for tok in ("GCM", "CHACHA20", "CCM", "POLY1305")) else 0.0 for c in ciphers],
            dtype=np.float32,
        )
    else:
        cipher_aead_flag = np.zeros(n, dtype=np.float32)

    cols: List[np.ndarray] = [
        no_ip_arr, a_count_log, aaaa_count_log, ip_div, ns_count_log,
        mx_count_arr, has_spf_arr, has_dmarc_arr, has_dkim_arr, dnssec_arr,
        ttl_a_log, ttl_ns_log, ttl_ratio,
        no_rdap_arr, domain_age_days_log, is_fresh_lt_30d, is_established_gt_365d,
        no_tls_arr, tls_cert_count_arr, cert_valid_len_log, tls_modern_proto, cipher_aead_flag,
    ]

    if include_has_subdomain:
        cols.append(_col_series("has_subdomain", False).astype(np.float32).to_numpy())

    if use_lexical:
        if "domain_body" in df.columns:
            bodies = [
                str(b) if (b is not None and not (isinstance(b, float) and math.isnan(b)) and str(b))
                else extract_domain_body(str(d))
                for b, d in zip(df["domain_body"].tolist(), df["domain"].tolist())
            ]
        else:
            bodies = [extract_domain_body(str(d)) for d in df["domain"].astype(str).tolist()]

        body_len = np.array([float(len(b)) for b in bodies], dtype=np.float32)
        body_entropy = np.array([compute_shannon_entropy(b) for b in bodies], dtype=np.float32)
        digit_ratio = np.array([sum(c.isdigit() for c in b) / max(len(b), 1) for b in bodies], dtype=np.float32)
        vowel_ratio = np.array([sum(c in "aeiou" for c in b.lower()) / max(len(b), 1) for b in bodies], dtype=np.float32)
        consonants = np.array([float(compute_longest_consonant_run(b)) for b in bodies], dtype=np.float32)
        hyphen_ratio = np.array([b.count("-") / max(len(b), 1) for b in bodies], dtype=np.float32)
        bigram_logp = np.array([compute_english_bigram_logprob(b) for b in bodies], dtype=np.float32)
        puny_brand = np.array([compute_punycode_or_brand_token(b) for b in bodies], dtype=np.float32)

        cols.extend([
            body_len, body_entropy, digit_ratio, vowel_ratio,
            consonants, hyphen_ratio, bigram_logp, puny_brand,
        ])

    return np.column_stack(cols).astype(np.float32)


def extract_domain_feature_vector(
    row: pd.Series,
    use_lexical: bool = True,
    include_has_subdomain: bool = False,
) -> np.ndarray:
    """Single-row wrapper matching `extract_domain_features_vectorized` 1:1."""
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return extract_domain_features_vectorized(
        pd.DataFrame([row_dict]),
        use_lexical=use_lexical,
        include_has_subdomain=include_has_subdomain,
    )[0]


# ---------------------------------------------------------------------------
# 4. Deterministic Label-Free Infrastructure Node Feature Extractors
# ---------------------------------------------------------------------------

def _hash_country_vec(country_str: Optional[str], dims: int = 4) -> List[float]:
    """Deterministic signed feature hashing for country codes (zero label leakage)."""
    if not country_str:
        return [0.0] * dims
    c = str(country_str).strip().upper()
    if not c:
        return [0.0] * dims
    digest = hashlib.blake2b(c.encode("utf-8"), digest_size=max(4, dims)).digest()
    return [1.0 if (digest[i] & 1) == 0 else -1.0 for i in range(dims)]


def build_ip_intrinsic_features(ip_to_id: Dict[str, int], ip_meta: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """Builds `[N_ip, 10]` intrinsic IP feature matrix."""
    n = len(ip_to_id)
    d = INFRA_INTRINSIC_DIMS["ip"]
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    mat = np.zeros((n, d), dtype=np.float32)
    for ip_str, idx in ip_to_id.items():
        meta = ip_meta.get(ip_str, {})
        is_v6 = 1.0 if ":" in ip_str else 0.0
        is_priv = 0.0
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            is_priv = 1.0 if (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved) else 0.0
        except ValueError:
            pass
        plen = meta.get("prefix_len")
        if plen is not None:
            plen_norm = float(plen) / (128.0 if is_v6 > 0.5 else 32.0)
        else:
            plen_norm = 0.0
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        cc = meta.get("country_code") or meta.get("country")
        geo_missing = 1.0 if (not cc and lat is None and lon is None) else 0.0
        lat_norm = float(lat) / 90.0 if lat is not None else 0.0
        lon_norm = float(lon) / 180.0 if lon is not None else 0.0
        c_hash = _hash_country_vec(cc, dims=4)
        mat[idx] = [
            is_v6, is_priv, plen_norm, geo_missing, lat_norm, lon_norm,
            c_hash[0], c_hash[1], c_hash[2], c_hash[3],
        ]
    return mat


_MAJOR_DNS_CDN_KW = (
    "cloudflare", "awsdns", "googledomains", "akam", "azure-dns", "fastly",
    "nsone", "ultradns", "domaincontrol", "registrar-servers", "dnsimple",
    "route53", "dnsmadeeasy", "dynect", "linode", "digitalocean", "hetzner",
)


def build_nameserver_intrinsic_features(ns_to_id: Dict[str, int], ns_meta: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """Builds `[N_ns, 8]` intrinsic nameserver feature matrix."""
    n = len(ns_to_id)
    d = INFRA_INTRINSIC_DIMS["nameserver"]
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    mat = np.zeros((n, d), dtype=np.float32)
    for ns_host, idx in ns_to_id.items():
        meta = ns_meta.get(ns_host, {})
        ns_low = ns_host.lower()
        ns_len = float(len(ns_low))
        ns_ent = compute_shannon_entropy(ns_low)
        is_major = 1.0 if any(kw in ns_low for kw in _MAJOR_DNS_CDN_KW) else 0.0
        glue_ips = meta.get("glue_ips") or set()
        glue_asns = meta.get("glue_asns") or set()
        glue_geos = meta.get("glue_countries") or set()
        orgs = " ".join(str(x).lower() for x in (meta.get("glue_orgs") or set()))
        is_cloud = 1.0 if any(kw in orgs for kw in ("cloud", "amazon", "aws", "google", "microsoft", "akamai", "fastly", "digitalocean", "hetzner", "ovh")) else 0.0
        tld_part = ns_low.rsplit(".", 1)[-1] if "." in ns_low else ""
        tld_is_cc = 1.0 if len(tld_part) == 2 and tld_part.isalpha() else 0.0
        mat[idx] = [
            ns_len,
            ns_ent,
            is_major,
            math.log1p(len(glue_ips)),
            math.log1p(len(glue_asns)),
            float(len(glue_geos)),
            is_cloud,
            tld_is_cc,
        ]
    return mat


_CLOUD_CDN_KW = ("cloudflare", "amazon", "aws", "google", "microsoft", "azure", "akamai", "fastly", "oracle", "alibaba", "tencent", "cdn", "cloud")
_HOSTING_VPS_KW = ("digitalocean", "hetzner", "ovh", "linode", "vultr", "hostinger", "contabo", "hosting", "vps", "server", "datacenter", "colocation", "dedicated", "rackspace")
_TELECOM_ISP_KW = ("telecom", "communications", "broadband", "wireless", "mobile", "cable", "fiber", "telefonica", "comcast", "verizon", "vodafone", "orange", "deutsche telekom", "isp", "telstra", "ntt")
_EDU_GOV_KW = ("university", "college", "institute", "school", "academic", "research", "government", "ministry", ".edu", ".gov")


def build_asn_intrinsic_features(asn_to_id: Dict[int, int], asn_meta: Dict[int, Dict[str, Any]]) -> np.ndarray:
    """Builds `[N_asn, 8]` intrinsic ASN feature matrix."""
    n = len(asn_to_id)
    d = INFRA_INTRINSIC_DIMS["asn"]
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    mat = np.zeros((n, d), dtype=np.float32)
    for asn_num, idx in asn_to_id.items():
        meta = asn_meta.get(asn_num, {})
        org_str = str(meta.get("as_org") or "").strip()
        org_low = org_str.lower()
        org_missing = 1.0 if not org_low else 0.0
        kw_cloud = 1.0 if any(k in org_low for k in _CLOUD_CDN_KW) else 0.0
        kw_host = 1.0 if any(k in org_low for k in _HOSTING_VPS_KW) else 0.0
        kw_tel = 1.0 if any(k in org_low for k in _TELECOM_ISP_KW) else 0.0
        kw_edu = 1.0 if any(k in org_low for k in _EDU_GOV_KW) else 0.0
        prefixes = meta.get("prefixes") or set()
        cc = meta.get("country")
        c_hash = _hash_country_vec(cc, dims=1)[0]
        mat[idx] = [
            math.log1p(max(0, int(asn_num))),
            org_missing,
            kw_cloud,
            kw_host,
            kw_tel,
            kw_edu,
            math.log1p(len(prefixes)),
            c_hash,
        ]
    return mat


_MAJOR_REGISTRAR_KW = (
    "godaddy", "namecheap", "tucows", "markmonitor", "csc", "cloudflare",
    "network solutions", "enom", "verisign", "gandi", "ovh", "ionos",
    "privacy", "whoisguard", "domains by proxy", "publicdomainregistry",
)


def build_registrar_intrinsic_features(reg_to_id: Dict[str, int], reg_meta: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """Builds `[N_reg, 6]` intrinsic registrar feature matrix."""
    n = len(reg_to_id)
    d = INFRA_INTRINSIC_DIMS["registrar"]
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    mat = np.zeros((n, d), dtype=np.float32)
    for reg_name, idx in reg_to_id.items():
        meta = reg_meta.get(reg_name, {})
        r_low = reg_name.lower()
        mat[idx] = [
            1.0 if meta.get("has_whois") else 0.0,
            1.0 if meta.get("has_email") else 0.0,
            1.0 if meta.get("has_url") else 0.0,
            1.0 if meta.get("has_tel") else 0.0,
            compute_shannon_entropy(r_low),
            1.0 if any(k in r_low for k in _MAJOR_REGISTRAR_KW) else 0.0,
        ]
    return mat


_FREE_AUTO_CA_KW = ("let's encrypt", "lets encrypt", "zerossl", "buypass", "google trust", "cloudflare", "cpanel")
_PAID_EV_CA_KW = ("digicert", "sectigo", "comodo", "globalsign", "entrust", "godaddy", "thawte", "geotrust", "symantec", "rapidssl")


def build_certificate_intrinsic_features(cert_to_id: Dict[str, int], cert_meta: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """Builds `[N_cert, 8]` intrinsic certificate feature matrix."""
    n = len(cert_to_id)
    d = INFRA_INTRINSIC_DIMS["certificate"]
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    mat = np.zeros((n, d), dtype=np.float32)
    for cert_key, idx in cert_to_id.items():
        meta = cert_meta.get(cert_key, {})
        v_len = float(meta.get("valid_len") or 0)
        v_len = max(0.0, v_len)
        is_short = 1.0 if (0.0 < v_len <= 90.0) else 0.0
        is_long = 1.0 if v_len > 365.0 else 0.0
        ext_cnt = float(meta.get("ext_count") or 0)
        org_str = str(meta.get("org") or "").strip().lower()
        org_missing = 1.0 if not org_str else 0.0
        is_free = 1.0 if any(k in org_str for k in _FREE_AUTO_CA_KW) else 0.0
        is_paid = 1.0 if any(k in org_str for k in _PAID_EV_CA_KW) else 0.0
        is_self = 1.0 if meta.get("is_self_signed") else 0.0
        mat[idx] = [
            math.log1p(v_len),
            is_short,
            is_long,
            ext_cnt,
            is_free,
            is_paid,
            org_missing,
            is_self,
        ]
    return mat


def build_prefix_intrinsic_features(prefix_to_id: Dict[str, int]) -> np.ndarray:
    """Builds `[N_prefix, 2]` intrinsic BGP prefix feature matrix."""
    n = len(prefix_to_id)
    d = INFRA_INTRINSIC_DIMS["prefix"]
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    mat = np.zeros((n, d), dtype=np.float32)
    for pfx_str, idx in prefix_to_id.items():
        is_v6 = 1.0 if ":" in pfx_str else 0.0
        plen_norm = 0.0
        if "/" in pfx_str:
            try:
                plen = float(pfx_str.split("/", 1)[1])
                plen_norm = plen / (128.0 if is_v6 > 0.5 else 32.0)
            except (ValueError, TypeError):
                plen_norm = 0.0
        mat[idx] = [is_v6, plen_norm]
    return mat


_MULTITENANT_APEX_SUFFIXES = (
    "github.io", "gitlab.io", "pages.dev", "vercel.app", "netlify.app", "herokuapp.com",
    "blogspot.com", "wordpress.com", "tumblr.com", "duckdns.org", "ddns.net", "no-ip.org",
    "amazonaws.com", "cloudfront.net", "azurewebsites.net", "appspot.com", "firebaseapp.com",
)
_LEGACY_GENERIC_TLDS = {"com", "net", "org", "edu", "gov", "mil", "int"}


def is_multitenant_public_suffix_apex(apex_domain: str) -> bool:
    """Returns True if apex belongs to a well-known shared multi-tenant / dynamic DNS service."""
    a = (apex_domain or "").strip().lower().rstrip(".")
    return any(a == s or a.endswith(f".{s}") for s in _MULTITENANT_APEX_SUFFIXES)


def build_apex_intrinsic_features(apex_to_id: Dict[str, int], apex_meta: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """Builds `[N_apex, 14]` intrinsic apex_domain feature matrix (RFC-001 total D_apex = 16 with split cols)."""
    n = len(apex_to_id)
    d = INFRA_INTRINSIC_DIMS["apex_domain"]
    if n == 0:
        return np.empty((0, d), dtype=np.float32)
    mat = np.zeros((n, d), dtype=np.float32)
    for apex_str, idx in apex_to_id.items():
        meta = apex_meta.get(apex_str, {})
        fanout = max(0, int(meta.get("subdomain_fanout_count") or 0))
        is_ps = 1.0 if (meta.get("is_public_suffix") or is_multitenant_public_suffix_apex(apex_str)) else 0.0
        no_rdap = 1.0 if meta.get("no_rdap", True) else 0.0
        age_days = max(0.0, float(meta.get("reg_age_days") or 0.0))
        has_age = bool(meta.get("has_reg_age", False))
        is_fresh = 1.0 if (has_age and age_days < 30.0) else 0.0
        is_est = 1.0 if (has_age and age_days > 365.0) else 0.0
        ns_cnt = max(0, int(meta.get("ns_count") or 0))
        has_w = 1.0 if meta.get("has_whois") else 0.0
        has_em = 1.0 if meta.get("has_email") else 0.0
        whois_stab = (0.4 * (1.0 - no_rdap)) + (0.3 * has_w) + (0.3 * has_em)
        tld = str(meta.get("tld") or (apex_str.rsplit(".", 1)[-1] if "." in apex_str else "")).lower()
        tld_g = 1.0 if tld in _LEGACY_GENERIC_TLDS else 0.0
        tld_cc = 1.0 if (len(tld) == 2 and tld.isalpha()) else 0.0
        tld_new = 1.0 if (tld and tld_g == 0.0 and tld_cc == 0.0) else 0.0
        mat[idx] = [
            math.log1p(fanout),
            is_ps,
            no_rdap,
            math.log1p(age_days),
            is_fresh,
            is_est,
            math.log1p(ns_cnt),
            has_w,
            has_em,
            whois_stab,
            tld_g,
            tld_cc,
            tld_new,
            float(len(apex_str)),
        ]
    return mat


def get_primary_domain_node_type(data: Any) -> str:
    """Returns `'fqdn'` for RFC-001 hierarchical graphs and `'domain'` otherwise."""
    node_types = getattr(data, "node_types", ())
    return "fqdn" if "fqdn" in node_types else "domain"
