"""Sunwa-AI — Pattern Analyzer.
Classifies incident clusters into pattern types:
  - Localized Incident
  - Complaint Surge
  - Recurring Infrastructure Issue
  - Distributed Systemic Issue
  - Potential Acute Hazard
  - Insufficient Evidence
"""
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _parse_ts(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt)
        except (ValueError, TypeError):
            pass
    return None


def classify_pattern(
    complaint_count: int,
    region_count: int,
    timestamps: List[str],
    has_safety_flags: bool,
    category: str,
    landmark_count: int = 1,
) -> str:
    """
    Classify the incident pattern from cluster characteristics.
    Returns one of the pattern type strings.
    """
    if complaint_count < 2:
        return "Localized Incident"

    parsed = [_parse_ts(ts) for ts in timestamps if _parse_ts(ts) is not None]

    if has_safety_flags and complaint_count >= 2:
        return "Potential Acute Hazard"

    if parsed and len(parsed) >= 2:
        sorted_ts = sorted(parsed)
        span_days = (sorted_ts[-1] - sorted_ts[0]).total_seconds() / 86400.0

        # High volume in short window → surge
        if complaint_count >= 4 and span_days <= 3:
            return "Complaint Surge"

        # Spread over weeks → recurring
        if span_days > 21 and complaint_count >= 3:
            return "Recurring Infrastructure Issue"

    # Multiple distinct locations / landmarks → systemic
    if landmark_count >= 3 or region_count >= 3:
        return "Distributed Systemic Issue"

    if complaint_count >= 2:
        return "Localized Incident"

    return "Insufficient Evidence"
