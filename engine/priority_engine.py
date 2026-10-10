"""Sunwa-AI — Deterministic Priority Engine.
Formula: P = 0.25*S + 0.20*A + 0.15*D + 0.25*R + 0.15*T
All weights are configurable. Safety override rule applied.
LLM never calculates or overrides priorities.
"""
import logging
from typing import List, Optional, Dict
from models.schemas import PriorityScore

logger = logging.getLogger(__name__)

# Default weights (sum must equal 1.0)
DEFAULT_WEIGHTS = {
    "severity": 0.25,
    "affected_pop": 0.20,
    "duration": 0.15,
    "safety_risk": 0.25,
    "recurrence": 0.15,
}

# Thresholds (configurable)
DEFAULT_THRESHOLDS = {
    "critical": 80,
    "high": 60,
    "medium": 40,
}

SAFETY_ESCALATION_THRESHOLD = 40  # If safety_risk_score >= this, escalate to at least High


def _priority_level(score: float, thresholds: Dict[str, int]) -> str:
    if score >= thresholds["critical"]:
        return "Critical"
    elif score >= thresholds["high"]:
        return "High"
    elif score >= thresholds["medium"]:
        return "Medium"
    return "Low"


def _severity_to_score(severity: str) -> float:
    mapping = {"High": 80.0, "Medium": 55.0, "Low": 25.0}
    return mapping.get(severity, 50.0)


def _complaint_count_to_affected(count: int) -> float:
    """Estimate affected population proxy from complaint count."""
    if count >= 20:
        return 90.0
    if count >= 10:
        return 75.0
    if count >= 5:
        return 60.0
    if count >= 3:
        return 45.0
    return 30.0


def _duration_to_score(span_days: float) -> float:
    if span_days >= 30:
        return 90.0
    if span_days >= 14:
        return 70.0
    if span_days >= 7:
        return 50.0
    if span_days >= 3:
        return 35.0
    return 20.0


def _safety_flags_to_score(safety_flags: List[str], pattern_type: str) -> float:
    if not safety_flags and pattern_type != "Potential Acute Hazard":
        return 0.0
    if pattern_type == "Potential Acute Hazard":
        return 90.0
    count = len(safety_flags)
    if count >= 3:
        return 80.0
    if count == 2:
        return 60.0
    return 40.0


def _recurrence_to_score(pattern_type: str, span_days: float, complaint_count: int) -> float:
    if pattern_type == "Recurring Infrastructure Issue":
        return 85.0
    if pattern_type == "Complaint Surge":
        return 60.0
    if pattern_type == "Distributed Systemic Issue":
        return 75.0
    if span_days > 7 and complaint_count >= 3:
        return 50.0
    return 20.0


def compute_priority(
    severity: str,
    complaint_count: int,
    span_days: float,
    safety_flags: List[str],
    pattern_type: str,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, int]] = None,
) -> PriorityScore:
    """Compute transparent numerical priority score with explanations."""
    w = weights or DEFAULT_WEIGHTS
    th = thresholds or DEFAULT_THRESHOLDS

    S = _severity_to_score(severity)
    A = _complaint_count_to_affected(complaint_count)
    D = _duration_to_score(span_days)
    R = _safety_flags_to_score(safety_flags, pattern_type)
    T = _recurrence_to_score(pattern_type, span_days, complaint_count)

    P = (w["severity"] * S + w["affected_pop"] * A + w["duration"] * D +
         w["safety_risk"] * R + w["recurrence"] * T)
    P = round(min(max(P, 0.0), 100.0), 1)

    level = _priority_level(P, th)

    # Safety escalation override
    override = False
    override_reason = ""
    if R >= SAFETY_ESCALATION_THRESHOLD and level == "Low":
        level = "Medium"
        override = True
        override_reason = f"Safety risk score {R:.0f} triggered escalation override to Medium."
    elif R >= SAFETY_ESCALATION_THRESHOLD and level == "Medium" and P < 60:
        level = "High"
        override = True
        override_reason = f"Safety risk score {R:.0f} with pattern '{pattern_type}' escalated to High."

    explanation = [
        f"Severity (S={S:.0f}, w={w['severity']:.0%}): {severity} complaint severity",
        f"Affected Population Proxy (A={A:.0f}, w={w['affected_pop']:.0%}): {complaint_count} related complaints",
        f"Duration (D={D:.0f}, w={w['duration']:.0%}): Issue spans {span_days:.0f} days",
        f"Safety Risk (R={R:.0f}, w={w['safety_risk']:.0%}): Safety indicators detected: {safety_flags or 'None'}",
        f"Recurrence (T={T:.0f}, w={w['recurrence']:.0%}): Pattern = '{pattern_type}'",
        f"Formula: P = {w['severity']:.2f}×{S:.0f} + {w['affected_pop']:.2f}×{A:.0f} + "
        f"{w['duration']:.2f}×{D:.0f} + {w['safety_risk']:.2f}×{R:.0f} + {w['recurrence']:.2f}×{T:.0f} = {P}",
    ]

    return PriorityScore(
        severity_score=S,
        affected_pop_score=A,
        duration_score=D,
        safety_risk_score=R,
        recurrence_score=T,
        overall_score=P,
        priority_level=level,
        safety_override_triggered=override,
        override_reason=override_reason,
        explanation_breakdown=explanation,
    )
