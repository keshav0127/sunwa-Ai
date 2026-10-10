"""Sunwa-AI — Complaint NLP Normalizer.
Handles: Unicode normalization, casing, script detection (Devanagari vs Latin),
category classification, severity/safety signal extraction, language detection.
Does NOT send data to any external API.
"""
import re
import unicodedata
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Pattern constants ────────────────────────────────────────────────────────

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[a-zA-Z]")

SEVERITY_HIGH_KEYWORDS = [
    r"\baccident\b", r"\bdanger(?:ous)?\b", r"\bhazard(?:ous)?\b",
    r"\burgent\b", r"\bcritical\b", r"\bimmediate\b", r"\bcrash\b",
    r"\bjivghena\b",  # Marathi: life-threatening
    r"\bजीवघेण\b", r"\bखतरनाक\b", r"\bखतरा\b",
    r"\bतात्काळ\b",  # Marathi: immediate
    r"\bturant\b", r"\bghabarahu\b",
]
SEVERITY_MEDIUM_KEYWORDS = [
    r"\bbroken\b", r"\bchoked\b", r"\bblocked\b", r"\bsubmerged\b",
    r"\bwaterlog(?:ging)?\b", r"\bpothole\b", r"\bkhadda\b", r"\bgaddha\b",
    r"\bleak(?:age)?\b", r"\boverflow\b", r"\bstench\b", r"\bsmell\b",
    r"\bखड्डा\b", r"\bगड्ढ\b", r"\bगटार\b", r"\bतुंबल\b",
    r"\bपाणी\b",  # water
    r"\bbreach\b", r"\bflood\b", r"\bcratre\b", r"\bcrater\b",
]

SAFETY_KEYWORDS = [
    r"\baccident\b", r"\bdeath\b", r"\binjur(?:y|ed|ies)\b",
    r"\bdanger(?:ous)?\b", r"\bhazard(?:ous)?\b",
    r"\bslipping\b", r"\bslip(?:ped)?\b", r"\bfall(?:ing)?\b",
    r"\bfire\b", r"\belectric(?:al)?\s*short\b", r"\belectric\s*shock\b",
    r"\bjivghena\b", r"\bजीवघेण\b", r"\bखतरनाक\b",
]

# Category keyword → category mapping (ordered: most specific first)
CATEGORY_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)\b(?:sewag|sewer|manhole|man-hole|chamber|septic|सांडपाणी|मॅनहोल|मैनहोल)\b", "Sewage & Manholes"),
    (r"(?i)\b(?:waterlog|water-log|water\s+log|flood|submerg|stormwater|drain|gutter|nullah|nala|nali|naali|गटार|तुंब|साच|नाली|नाला|ओसंडून|डबके)\b", "Drainage & Flooding"),
    (r"(?i)\b(?:pothol|pot\s*hol|khadd|gaddh|crater|asphalt|tar\s*road|road.*broken|broken.*road|road.*damag|damaged.*road|cave-?in|uneven\s*road|खड्डा|खड्डे|गड्डा|गड्ढे|रस्त)\b", "Road Damage & Potholes"),
    (r"(?i)\b(?:streetlight|street\s*light|lamppost|darkness|pitch\s*dark|no\s*light|light.*off|electricity|power\s*cut|transformer|wire|sparking|short\s*circuit|batti|andhar|andhera|bijli|khamba|स्ट्रीटला|लाईट|लाइट|अंधार|विजेचा|बत्ती|खांब|विद्युत)\b", "Electrical & Streetlights"),
    (r"(?i)\b(?:garbag|waste|dump|litter|trash|rubbish|refuse|kachra|kachada|kudha|kuda|dher|sanitat|dustbin|solid\s*waste|stray\s*cattle|stench|foul\s*smell|badbu|badboo|कचरा|कूड़ा|घाण|दुर्गंधी)\b", "Solid Waste & Sanitation"),
    (r"(?i)\b(?:water\s*supply|pipeline|pipe\s*line|pipe\s*burst|burst\s*pipe|water\s*leak|leakage|leak|no\s*water|low\s*pressure|water\s*shortage|drinking\s*water|dirty\s*water|tanker|main\s*line|paani|pani|पाणी|पाण्याची|पाणीपुरवठा|गळती|पाइपलाइन|पाईप)\b", "Water Supply & Pipeline"),
    (r"(?i)\b(?:footpath|foot\s*path|pavement|sidewalk|walkway|paver\s*block|pedestrian|broken\s*tile|curb|kerb|पदपथ|पाऊलवाट|पेव्हर\s*ब्लॉक|फूटपाथ)\b", "Footpath & Pavement"),
    (r"(?i)\b(?:tree|branch|fallen\s*tree|horticulture|pruning|jhad|ped|झाड|झाडे|फांदी|फांद्या)\b", "Tree / Vegetation"),
]


def detect_language(text: str) -> str:
    """Simple heuristic language detection: deva, en, or hi-romanized."""
    if _DEVANAGARI_RE.search(text):
        # Could be hi or mr, label generically 'deva'
        return "deva"
    words = text.lower().split()
    romanized_markers = {"ke", "ka", "hai", "mein", "par", "bada", "khadda", "nahi",
                         "ahe", "aahe", "cha", "chi", "che", "gaddha", "rastyavar"}
    if romanized_markers & set(words):
        return "romanized"
    return "en"


def normalize_text(text: str) -> str:
    """Unicode normalization + Latin lowercasing, preserving Devanagari script."""
    # NFC normalization for consistent Unicode
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize punctuation (keep key separators)
    text = re.sub(r"[।|]", " ", text)
    text = re.sub(r"[''`]", "'", text)
    # Lowercase only ASCII/Latin characters
    def _lowcase(m: re.Match) -> str:
        return m.group().lower()
    text = re.sub(r"[a-zA-Z]+", _lowcase, text)
    return text


def classify_category(text: str) -> str:
    """Return primary issue category from complaint text."""
    lower = text.lower()
    for pattern, category in CATEGORY_PATTERNS:
        if re.search(pattern, lower):
            return category
    return "General Infrastructure"


def extract_severity(text: str) -> Tuple[str, List[str]]:
    """Return severity level and matched indicator keywords."""
    lower = text.lower()
    indicators: List[str] = []
    for pat in SEVERITY_HIGH_KEYWORDS:
        if re.search(pat, lower):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                indicators.append(m.group())
    if indicators:
        return "High", list(set(indicators))

    for pat in SEVERITY_MEDIUM_KEYWORDS:
        if re.search(pat, lower):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                indicators.append(m.group())
    if indicators:
        return "Medium", list(set(indicators))

    return "Low", []


def extract_safety_flags(text: str) -> List[str]:
    """Extract safety related keywords citing imminent risk."""
    lower = text.lower()
    flags: List[str] = []
    for pat in SAFETY_KEYWORDS:
        if re.search(pat, lower):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                flags.append(m.group().strip())
    return list(set(flags))
