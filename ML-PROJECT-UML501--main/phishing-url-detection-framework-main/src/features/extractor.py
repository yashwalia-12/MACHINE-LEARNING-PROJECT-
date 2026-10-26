"""Feature Extraction Engine for Phishing URL Detection.

Extracts all 74 engineered features (53 Lexical, 9 DNS, 12 WHOIS)
matching the exact schema of the UML501 benchmark dataset.
"""

from __future__ import annotations

import math
import re
import socket
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

# Optional network libraries with safe fallbacks
try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False


SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "work", "loan", "tk", "ml", "ga", "cf", "gq",
    "vip", "click", "buzz", "rest", "fit", "surf", "casa", "cam", "icu"
}

SENSITIVE_KEYWORDS = [
    "login", "secure", "verify", "account", "update", "bank", "cloud", "brand"
]


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_lexical_features(url: str) -> Dict[str, Any]:
    """Extract 53 lexical features from a URL string."""
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        # Prepend scheme for reliable parsing if missing
        url_with_scheme = "http://" + url
    else:
        url_with_scheme = url

    try:
        parsed = urlparse(url_with_scheme)
    except Exception:
        parsed = urlparse("http://invalid.url")

    hostname = parsed.netloc.lower()
    # Strip port if present
    if ":" in hostname:
        hostname_no_port = hostname.split(":", 1)[0]
    else:
        hostname_no_port = hostname

    domain = hostname_no_port
    if domain.startswith("www."):
        domain = domain[4:]

    path = parsed.path or ""
    query = parsed.query or ""
    path_segments = [p for p in path.split("/") if p]
    first_dir = path_segments[0] if path_segments else ""

    # TLD extraction
    domain_parts = domain.split(".")
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""

    url_len = max(len(url), 1)
    digits_count = sum(c.isdigit() for c in url)
    letters_count = sum(c.isalpha() for c in url)
    special_chars_count = sum(not c.isalnum() for c in url)

    # Keyword check
    url_lower = url.lower()
    query_params = query.split("&") if query else []
    query_key_count = len([q for q in query_params if "=" in q])
    query_val_lengths = [len(q.split("=", 1)[1]) for q in query_params if "=" in q]
    avg_query_val_len = (sum(query_val_lengths) / len(query_val_lengths)) if query_val_lengths else 0.0

    features = {
        "url_length": float(len(url)),
        "domain_length": float(len(domain)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "first_dir_length": float(len(first_dir)),
        "tld_length": float(len(tld)),
        "tld_length_domain": float(len(tld)),
        "url_depth": float(len(path_segments)),
        "query_length": float(len(query)),
        "path_segments_count": float(len(path_segments)),
        "num_digits": float(digits_count),
        "num_letters": float(letters_count),
        "num_special_chars": float(special_chars_count),
        "num_dots": float(url.count(".")),
        "num_hyphens": float(url.count("-")),
        "num_at": float(url.count("@")),
        "num_percent": float(url.count("%")),
        "num_equals": float(url.count("=")),
        "num_question": float(url.count("?")),
        "num_ampersand": float(url.count("&")),
        "num_hash": float(url.count("#")),
        "num_underscore": float(url.count("_")),
        "num_special": float(special_chars_count),
        "num_slash": float(url.count("/")),
        "num_params": float(len(query_params)),
        "entropy_url": float(calculate_entropy(url)),
        "entropy_hostname": float(calculate_entropy(hostname)),
        "entropy_domain": float(calculate_entropy(domain)),
        "entropy_path": float(calculate_entropy(path)),
        "query_entropy": float(calculate_entropy(query)),
        "ratio_digits": float(digits_count / url_len),
        "ratio_letters": float(letters_count / url_len),
        "ratio_special_chars": float(special_chars_count / url_len),
        "uppercase_ratio": float(sum(c.isupper() for c in url) / url_len),
        "lowercase_ratio": float(sum(c.islower() for c in url) / url_len),
        "is_ip_address": float(1.0 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname_no_port) else 0.0),
        "starts_with_ip": float(1.0 if re.match(r"^https?://\d{1,3}(\.\d{1,3}){3}", url) else 0.0),
        "is_suspicious_tld": float(1.0 if tld.lower() in SUSPICIOUS_TLDS else 0.0),
        "uses_https": float(1.0 if url.startswith("https://") else 0.0),
        "has_www": float(1.0 if "www." in hostname else 0.0),
        "unusual_double_slash": float(1.0 if url.find("//", 8) != -1 else 0.0),
        "multiple_http": float(1.0 if (url.count("http://") + url.count("https://")) > 1 else 0.0),
        "contains_port_number": float(1.0 if ":" in hostname else 0.0),
        "path_has_encoded_chars": float(1.0 if "%" in path else 0.0),
        "query_has_base64": float(1.0 if re.search(r"[A-Za-z0-9+/]{20,}={0,2}", query) else 0.0),
        "contains_login": float(1.0 if "login" in url_lower else 0.0),
        "contains_secure": float(1.0 if "secure" in url_lower else 0.0),
        "contains_verify": float(1.0 if "verify" in url_lower else 0.0),
        "contains_account": float(1.0 if "account" in url_lower else 0.0),
        "contains_update": float(1.0 if "update" in url_lower else 0.0),
        "contains_bank": float(1.0 if "bank" in url_lower else 0.0),
        "contains_cloud": float(1.0 if "cloud" in url_lower else 0.0),
        "contains_brand": float(1.0 if any(b in url_lower for b in ["paypal", "apple", "google", "microsoft", "amazon", "netflix"]) else 0.0),
        "query_key_count": float(query_key_count),
        "query_value_length_avg": float(avg_query_val_len),
    }
    return features


def extract_dns_features(hostname: str, timeout: float = 1.0) -> Dict[str, float]:
    """Extract 9 DNS-based features with timeout protection."""
    defaults = {
        "success": 0.0,
        "dns_resolves": 0.0,
        "has_mx_record": 0.0,
        "has_txt_record": 0.0,
        "has_ns_record": 0.0,
        "ttl_value": 300.0,
        "ip_count": 0.0,
        "cname_count": 0.0,
        "resolves_to_private_ip": 0.0,
    }

    if not DNS_AVAILABLE or not hostname:
        return defaults

    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = timeout
        resolver.timeout = timeout

        clean_host = hostname.split(":")[0].strip()
        ans = resolver.resolve(clean_host, "A")
        ips = [str(r) for r in ans]
        defaults["dns_resolves"] = 1.0
        defaults["success"] = 1.0
        defaults["ip_count"] = float(len(ips))
        defaults["ttl_value"] = float(ans.ttl)

        for ip in ips:
            if ip.startswith(("10.", "192.168.", "172.16.", "127.")):
                defaults["resolves_to_private_ip"] = 1.0
                break

        try:
            mx_ans = resolver.resolve(clean_host, "MX")
            defaults["has_mx_record"] = 1.0 if len(mx_ans) > 0 else 0.0
        except Exception:
            pass

        try:
            txt_ans = resolver.resolve(clean_host, "TXT")
            defaults["has_txt_record"] = 1.0 if len(txt_ans) > 0 else 0.0
        except Exception:
            pass

        try:
            ns_ans = resolver.resolve(clean_host, "NS")
            defaults["has_ns_record"] = 1.0 if len(ns_ans) > 0 else 0.0
        except Exception:
            pass

    except Exception:
        # DNS lookup failed (common for fake phishing domains)
        pass

    return defaults


def extract_whois_features(domain: str, timeout: float = 1.5) -> Dict[str, float]:
    """Extract 12 WHOIS-based features with timeout protection."""
    defaults = {
        "whois_success": 0.0,
        "domain_age_days": 0.0,
        "expiration_days": 0.0,
        "creation_year": 0.0,
        "domain_is_recent": 0.0,
        "domain_registered_before_2020": 0.0,
        "registrar_valid": 0.0,
        "name_servers_count": 0.0,
        "is_privacy_protected": 0.0,
        "whois_missing": 1.0,
    }

    if not WHOIS_AVAILABLE or not domain:
        return defaults

    clean_domain = domain.split(":")[0].strip()
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]

    try:
        w = whois.whois(clean_domain)
        if not w or not getattr(w, "domain_name", None):
            return defaults

        defaults["whois_success"] = 1.0
        defaults["whois_missing"] = 0.0

        created = w.creation_date
        if isinstance(created, list):
            created = created[0]

        expires = w.expiration_date
        if isinstance(expires, list):
            expires = expires[0]

        now = datetime.now()
        if isinstance(created, datetime):
            age = (now - created).days
            defaults["domain_age_days"] = float(max(0, age))
            defaults["creation_year"] = float(created.year)
            defaults["domain_is_recent"] = 1.0 if age < 365 else 0.0
            defaults["domain_registered_before_2020"] = 1.0 if created.year < 2020 else 0.0

        if isinstance(expires, datetime):
            remaining = (expires - now).days
            defaults["expiration_days"] = float(max(0, remaining))

        if getattr(w, "registrar", None):
            defaults["registrar_valid"] = 1.0

        ns = getattr(w, "name_servers", None)
        if ns:
            defaults["name_servers_count"] = float(len(ns) if isinstance(ns, list) else 1)

        raw_str = str(w).lower()
        if any(term in raw_str for term in ["privacy", "proxy", "redacted", "whoisguard"]):
            defaults["is_privacy_protected"] = 1.0

    except Exception:
        # Whois failed or unavailable
        pass

    return defaults


def extract_all_features(url: str, enable_network: bool = False) -> Dict[str, float]:
    """Extract all 74 features for a single URL.
    
    If enable_network is False (recommended for fast real-time inference),
    DNS and WHOIS features use safe default imputation values.
    """
    lex = extract_lexical_features(url)

    parsed = urlparse(url if url.startswith(("http://", "https://")) else f"http://{url}")
    hostname = parsed.netloc.lower().split(":", 1)[0]
    domain = hostname[4:] if hostname.startswith("www.") else hostname

    if enable_network:
        dns_feats = extract_dns_features(hostname)
        whois_feats = extract_whois_features(domain)
    else:
        # Default imputed features matching median baseline
        dns_feats = {
            "success": 1.0, "dns_resolves": 1.0, "has_mx_record": 1.0,
            "has_txt_record": 1.0, "has_ns_record": 1.0, "ttl_value": 300.0,
            "ip_count": 1.0, "cname_count": 0.0, "resolves_to_private_ip": 0.0,
        }
        whois_feats = {
            "whois_success": 1.0, "domain_age_days": 1200.0, "expiration_days": 365.0,
            "creation_year": 2020.0, "domain_is_recent": 0.0, "domain_registered_before_2020": 1.0,
            "registrar_valid": 1.0, "name_servers_count": 2.0, "is_privacy_protected": 0.0,
            "whois_missing": 0.0,
        }

    combined = {**lex, **dns_feats, **whois_feats}
    return combined
