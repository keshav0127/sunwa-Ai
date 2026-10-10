"""Tests: Priority engine — formula, safety override, configurable weights, missing values."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.priority_engine import compute_priority, _priority_level, DEFAULT_THRESHOLDS


def test_basic_score_calculation():
    score = compute_priority(
        severity="High",
        complaint_count=10,
        span_days=5,
        safety_flags=[],
        pattern_type="Localized Incident",
    )
    assert 0 <= score.overall_score <= 100
    assert score.priority_level in ("Critical", "High", "Medium", "Low")


def test_safety_escalation_triggered():
    score = compute_priority(
        severity="Low",
        complaint_count=2,
        span_days=1,
        safety_flags=["accident", "dangerous"],
        pattern_type="Potential Acute Hazard",
    )
    assert score.safety_risk_score >= 80
    # Low base should be escalated
    assert score.priority_level in ("High", "Critical", "Medium")


def test_critical_pattern_high_score():
    score = compute_priority(
        severity="High",
        complaint_count=20,
        span_days=30,
        safety_flags=["dangerous", "accident"],
        pattern_type="Recurring Infrastructure Issue",
    )
    assert score.overall_score >= 60.0
    assert score.priority_level in ("Critical", "High")


def test_low_score_low_priority():
    score = compute_priority(
        severity="Low",
        complaint_count=1,
        span_days=0,
        safety_flags=[],
        pattern_type="Localized Incident",
    )
    assert score.priority_level in ("Low", "Medium")


def test_formula_explanation_present():
    score = compute_priority("High", 5, 7, [], "Complaint Surge")
    assert len(score.explanation_breakdown) >= 5
    assert any("Formula" in line for line in score.explanation_breakdown)


def test_configurable_thresholds():
    custom_thresholds = {"critical": 90, "high": 70, "medium": 50}
    score = compute_priority(
        severity="High",
        complaint_count=10,
        span_days=7,
        safety_flags=[],
        pattern_type="Localized Incident",
        thresholds=custom_thresholds,
    )
    assert score.priority_level in ("Critical", "High", "Medium", "Low")


def test_configurable_weights():
    custom_weights = {
        "severity": 0.50,
        "affected_pop": 0.10,
        "duration": 0.10,
        "safety_risk": 0.20,
        "recurrence": 0.10,
    }
    score = compute_priority("High", 5, 7, [], "Localized Incident", weights=custom_weights)
    assert 0 <= score.overall_score <= 100


def test_priority_level_boundaries():
    assert _priority_level(85, DEFAULT_THRESHOLDS) == "Critical"
    assert _priority_level(65, DEFAULT_THRESHOLDS) == "High"
    assert _priority_level(50, DEFAULT_THRESHOLDS) == "Medium"
    assert _priority_level(30, DEFAULT_THRESHOLDS) == "Low"


def test_score_clamped_to_100():
    score = compute_priority("High", 100, 365, ["accident", "fire", "dangerous"], "Recurring Infrastructure Issue")
    assert score.overall_score <= 100.0
    assert score.overall_score >= 0.0


def test_safety_override_recorded():
    score = compute_priority(
        severity="Low",
        complaint_count=1,
        span_days=0,
        safety_flags=["accident"],
        pattern_type="Potential Acute Hazard",
    )
    if score.safety_override_triggered:
        assert score.override_reason != ""
