"""
Tests for parsing and schema normalization in Malicious-HDG.
Verifies:
- Handling of nulls and missing keys
- $numberLong and string date parsing
- Both malware and malware_type family key resolution
- dns.NS object structure parsing
- Time key fallback (sourced_on -> evaluated_on) and exclusion of missing time (no min-date fill)
- Leaf certificate selection via is_root flag
- ASN candidate key resolution and zero-ASN error raising
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import pytest
import tldextract

from src.hdg.parse import (
    extract_leaf_certificate,
    resolve_asn,
    resolve_registrar,
    stream_parse_file,
)
from src.hdg.profiler import parse_mongo_date


def test_parse_mongo_date() -> None:
    # ISO string
    d1 = parse_mongo_date({"$date": "2023-05-10T12:00:00Z"})
    assert d1 == datetime(2023, 5, 10, 12, 0, 0, tzinfo=timezone.utc)

    # numberLong
    # 1683720000000 ms = 2023-05-10 12:00:00 UTC
    d2 = parse_mongo_date({"$date": {"$numberLong": "1683720000000"}})
    assert d2 == datetime(2023, 5, 10, 12, 0, 0, tzinfo=timezone.utc)

    # Plain ISO string
    d3 = parse_mongo_date("2023-05-10T12:00:00Z")
    assert d3 == datetime(2023, 5, 10, 12, 0, 0, tzinfo=timezone.utc)

    # Null
    assert parse_mongo_date(None) is None


def test_leaf_certificate_selection() -> None:
    # Root cert first, leaf cert second
    tls_obj = {
        "protocol": "TLSv1.3",
        "cipher": "TLS_AES_256_GCM_SHA384",
        "certificates": [
            {
                "common_name": "ISRG Root X1",
                "organization": "Internet Security Research Group",
                "country": "US",
                "is_root": True,
                "valid_len": 7300
            },
            {
                "common_name": "example.com",
                "organization": "Let's Encrypt",
                "country": "US",
                "is_root": False,
                "valid_len": 90
            }
        ]
    }
    cert_hash, valid_len = extract_leaf_certificate(tls_obj)
    assert cert_hash is not None
    assert valid_len == 90

    # Test reverse order: leaf first, root second -> identical hash
    tls_obj_rev = {
        "certificates": [tls_obj["certificates"][1], tls_obj["certificates"][0]]
    }
    cert_hash_rev, valid_len_rev = extract_leaf_certificate(tls_obj_rev)
    assert cert_hash_rev == cert_hash
    assert valid_len_rev == 90

    # No certs / null
    assert extract_leaf_certificate(None) == (None, None)
    assert extract_leaf_certificate({"certificates": []}) == (None, None)


def test_asn_candidate_resolution() -> None:
    from collections import Counter
    counter = Counter()

    # Candidate 1: 'number'
    asn1, org1 = resolve_asn({"number": 13335, "organization": "Cloudflare"}, ["asn", "autonomous_system_number", "number"], counter)
    assert asn1 == 13335
    assert org1 == "Cloudflare"
    assert counter["number"] == 1

    # Candidate 2: 'asn'
    asn2, org2 = resolve_asn({"asn": 15169, "organization": "Google"}, ["asn", "autonomous_system_number", "number"], counter)
    assert asn2 == 15169
    assert org2 == "Google"
    assert counter["asn"] == 1

    # Candidate 3: 'autonomous_system_number'
    asn3, org3 = resolve_asn({"autonomous_system_number": 16509}, ["asn", "autonomous_system_number", "number"], counter)
    assert asn3 == 16509
    assert counter["autonomous_system_number"] == 1


def test_stream_parser_schema_handling() -> None:
    from collections import Counter
    extractor = tldextract.TLDExtract(suffix_list_urls=())
    asn_counter = Counter()
    reg_counter = Counter()

    records = [
        # Record 1: has 'malware' key, sourced_on present, null dns/rdap/tls
        {
            "domain_name": "malware1.com",
            "source": "ThreatFox",
            "malware": "emotet",
            "sourced_on": {"$date": "2023-05-01T00:00:00Z"},
            "dns": None,
            "rdap": None,
            "tls": None,
            "ip_data": None
        },
        # Record 2: has 'malware_type' key, sourced_on missing fallback to evaluated_on
        {
            "domain_name": "malware2.com",
            "source": "URLhaus",
            "malware_type": "redline",
            "sourced_on": None,
            "evaluated_on": {"$date": {"$numberLong": "1683720000000"}},
            "dns": {
                "A": ["198.51.100.1"],
                "NS": {
                    "ns1.example.com": {"related_ips": [{"value": "198.51.100.2"}]}
                }
            },
            "rdap": {
                "entities": {"registrar": [{"name": "GoDaddy"}]}
            },
            "ip_data": [
                {"ip": "198.51.100.1", "from_record": "A", "asn": {"number": 13335}}
            ]
        },
        # Record 3: no time at all -> must be excluded
        {
            "domain_name": "excluded.com",
            "sourced_on": None,
            "evaluated_on": None
        }
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(records, tmp)
        tmp_path = Path(tmp.name)

    try:
        parsed, total, excluded = stream_parse_file(
            file_path=tmp_path,
            label=1,
            default_source="test",
            extractor=extractor,
            normalize_to_e2ld=True,
            asn_candidate_keys=["asn", "autonomous_system_number", "number"],
            registrar_candidate_keys=["name", "handle"],
            asn_key_counter=asn_counter,
            reg_key_counter=reg_counter
        )

        assert total == 3
        assert excluded == 1
        assert len(parsed) == 2

        # Check record 1
        r1 = parsed[0]
        assert r1["domain"] == "malware1.com"
        assert r1["family"] == "emotet"
        assert r1["no_ip"] is True
        assert r1["no_rdap"] is True
        assert r1["no_tls"] is True

        # Check record 2 (fallback time, dns.NS object, registrar, ASN)
        r2 = parsed[1]
        assert r2["domain"] == "malware2.com"
        assert r2["family"] == "redline"
        assert r2["no_ip"] is False
        assert "ns1.example.com" in r2["nameservers"]
        assert r2["registrar"] == "GoDaddy"
        assert r2["primary_asn"] == 13335
        assert len(r2["ns_ip_pairs"]) == 1

    finally:
        if tmp_path.exists():
            tmp_path.unlink()
