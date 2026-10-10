"""Sunwa-AI — Intervention Manager.
Loads intervention records and computes before/after complaint activity.
Results labelled observational — correlation, not causation.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

from models.schemas import InterventionRecord

logger = logging.getLogger(__name__)


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s)[:19], fmt)
        except (ValueError, TypeError):
            pass
    return None


def compute_before_after(
    intervention: InterventionRecord,
    all_complaints,  # List[EnrichedComplaint]
    window_days: int = 14,
) -> Dict[str, Any]:
    """
    Calculate complaint counts before and after the intervention.
    Uses intervention created_timestamp as the split point.
    Returns observational data only — explicitly labelled.
    """
    split_ts = _parse_ts(intervention.created_timestamp)
    if split_ts is None:
        return {
            "error": "Intervention timestamp unavailable — cannot compute before/after.",
            "intervention_id": intervention.intervention_id,
        }

    before_start = split_ts - timedelta(days=window_days)
    after_end = split_ts + timedelta(days=window_days)

    target_cid = intervention.target_cluster_id.lower()
    region = intervention.region.lower()

    # Filter for same region complaints
    region_complaints = [
        c for c in all_complaints
        if c.region.strip().lower() == region
    ]

    before_count = 0
    after_count = 0
    before_ids = []
    after_ids = []

    for c in region_complaints:
        # Match by cluster_id or region proximity
        ts = _parse_ts(c.submitted_at)
        if ts is None:
            continue
        if before_start <= ts < split_ts:
            before_count += 1
            before_ids.append(c.complaint_id)
        elif split_ts <= ts <= after_end:
            after_count += 1
            after_ids.append(c.complaint_id)

    if before_count == 0:
        pct_change = None
        pct_label = "N/A (No baseline complaints in window)"
    else:
        pct_change = round(((after_count - before_count) / before_count) * 100.0, 1)
        pct_label = f"{'+' if pct_change >= 0 else ''}{pct_change}%"

    return {
        "intervention_id": intervention.intervention_id,
        "region": intervention.region,
        "split_timestamp": split_ts.isoformat(),
        "window_days": window_days,
        "before_count": before_count,
        "after_count": after_count,
        "before_complaint_ids": before_ids[:10],
        "after_complaint_ids": after_ids[:10],
        "pct_change": pct_change,
        "pct_label": pct_label,
        "observation_disclaimer": (
            "Observational data only. A change in complaint volume after an intervention "
            "does not imply causation. Other factors may have contributed."
        ),
        "is_simulated": intervention.is_simulated,
    }
