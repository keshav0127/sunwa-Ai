"""Sunwa-AI — Groq LLM Client.
Uses GROQ_API env variable. Configured model: openai/gpt-oss-20b.
Invoked ONLY as location-resolver fallback — never for scoring or clustering.
Gracefully disabled on missing/invalid key without crashing.
"""
import os
import logging
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from groq import Groq  # type: ignore
except ImportError:
    Groq = None  # type: ignore

_ENV_KEYS_TO_TRY = ["GROQ_API", "GROQ_API_KEY", "GROK_API"]
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1"

# Runtime flag — set to True on auth failure, preventing retry loops
_API_DISABLED = False
_API_KEY: Optional[str] = None
_KEY_PRESENT: bool = False


def _load_key() -> Optional[str]:
    global _API_KEY, _KEY_PRESENT
    for var in _ENV_KEYS_TO_TRY:
        val = os.environ.get(var, "").strip()
        if val:
            _API_KEY = val
            _KEY_PRESENT = True
            return val
    _API_KEY = None
    _KEY_PRESENT = False
    return None


def get_api_status() -> dict:
    """Returns API availability status without exposing the key value."""
    key = _load_key()
    return {
        "key_present": bool(key),
        "api_disabled": _API_DISABLED,
        "model": GROQ_MODEL,
        "env_var_tried": _ENV_KEYS_TO_TRY,
    }


def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "invalid api key", "incorrect api key", "authentication",
        "unauthorized", "401", "403", "forbidden",
    ])


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ["rate limit", "429", "too many requests"])


def resolve_location_with_llm(
    complaint_text: str,
    region: str,
    candidates: List[str],
) -> Optional[str]:
    """Ask Groq to suggest ONE landmark from the candidates list.
    Returns the suggested landmark name string, or None on any failure.
    NEVER invents new landmarks or coordinates.
    """
    global _API_DISABLED

    if _API_DISABLED:
        return None

    api_key = _load_key()
    if not api_key:
        logger.info("GROQ_API not configured — skipping LLM fallback.")
        return None

    if not candidates:
        return None

    if Groq is None:
        logger.warning("groq package not installed — LLM fallback disabled.")
        _API_DISABLED = True
        return None

    candidates_list = "\n".join(f"- {c}" for c in candidates[:20])
    prompt = (
        f"You are a location identification assistant for a civic grievance system in Navi Mumbai.\n"
        f"Citizen complaint: \"{complaint_text}\"\n"
        f"Administrative region: {region}\n\n"
        f"From the EXACT list below, identify the ONE landmark most likely referenced in the complaint.\n"
        f"If no landmark fits, respond with NONE.\n"
        f"Respond with ONLY the exact landmark name from the list, nothing else.\n\n"
        f"Available landmarks:\n{candidates_list}"
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
        )
        suggestion = response.choices[0].message.content.strip()
        if suggestion.upper() == "NONE" or not suggestion:
            return None
        # Only return if suggestion is in candidate list (exact or partial)
        for c in candidates:
            if suggestion.lower() == c.lower():
                return c
        logger.warning("LLM suggestion '%s' not in candidate list — rejected.", suggestion)
        return None

    except Exception as exc:
        if _is_auth_error(exc):
            _API_DISABLED = True
            logger.warning(
                "Groq authentication failed (invalid API key). LLM fallback disabled. "
                "Check GROQ_API environment variable."
            )
        elif _is_rate_limit(exc):
            logger.warning("Groq rate limit hit. Will retry next eligible request.")
        else:
            logger.error("Groq API error: %s", type(exc).__name__)
        return None


def interpret_cluster_context_with_llm(
    cluster_id: str,
    landmark_name: str,
    region: str,
    category: str,
    pattern_type: str,
    deterministic_score: float,
    priority_level: str,
    complaint_records: List[dict],
) -> Optional[str]:
    """Ask Groq to provide a grounded contextual interpretation for an already-prioritized incident.
    The LLM does NOT calculate or change the deterministic score.
    Cites only actual complaint IDs provided. Returns grounded text or None on failure.
    """
    global _API_DISABLED

    if _API_DISABLED:
        return None

    api_key = _load_key()
    if not api_key:
        return None

    if Groq is None:
        _API_DISABLED = True
        return None

    if not complaint_records:
        return None

    excerpts_str = "\n".join(
        f"- [{c['complaint_id']}] ({str(c.get('submitted_at', ''))[:10]}): \"{c['text']}\""
        for c in complaint_records[:10]
    )
    valid_ids = [c["complaint_id"] for c in complaint_records[:3]]

    prompt = (
        f"You are a civic intelligence analyst interpreting an already prioritized municipal grievance cluster in {region}, Navi Mumbai.\n\n"
        f"CLUSTER METADATA:\n"
        f"- Cluster ID: {cluster_id}\n"
        f"- Landmark: {landmark_name}\n"
        f"- Category: {category}\n"
        f"- Pattern: {pattern_type}\n"
        f"- Deterministic Priority Score: {deterministic_score}/100 ({priority_level})\n\n"
        f"OFFICIAL GROUNDED COMPLAINTS ({len(complaint_records)} total):\n"
        f"{excerpts_str}\n\n"
        f"CRITICAL RULES:\n"
        f"1. DO NOT change, recalculate, or override the deterministic priority score ({deterministic_score}).\n"
        f"2. Reference ONLY the actual complaint IDs listed above (e.g. {valid_ids}). NEVER invent complaint IDs.\n"
        f"3. Do not fabricate facts, injuries, population numbers, or quotes not present in the complaints.\n\n"
        f"Provide a structured analysis with these exact headings:\n"
        f"• CONTEXTUAL SUMMARY: (Synthesize the recurring issue from the complaints)\n"
        f"• GROUNDED EVIDENCE: (Cite specific complaint IDs and direct quotes demonstrating severity)\n"
        f"• OPERATIONAL GUIDANCE: (Field recommendations for the municipal response team)"
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as exc:
        if _is_auth_error(exc):
            _API_DISABLED = True
            logger.warning("Groq authentication failed during cluster interpretation. LLM disabled.")
        elif _is_rate_limit(exc):
            logger.warning("Groq rate limit hit during cluster interpretation.")
        else:
            logger.error("Groq interpretation error: %s", type(exc).__name__)
        return None
