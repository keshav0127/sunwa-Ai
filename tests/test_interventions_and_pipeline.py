"""Sunwa-AI — Tests for Interventions, Pipeline Submission, and LLM Contextual Interpretation."""
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime

from models.schemas import InterventionRecord, RawComplaint, EnrichedComplaint, ResolvedLocation, PriorityScore, IncidentCluster, ActionRecommendation
from engine.data_loader import save_intervention_append, load_interventions
from engine.intervention_manager import compute_before_after
from engine.llm_client import interpret_cluster_context_with_llm
from engine.pipeline import CivicPulsePipeline


def test_save_and_load_interventions():
    """Verify that interventions persist to JSON and load back accurately."""
    with TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_interventions.json"

        rec1 = InterventionRecord(
            intervention_id="INT-2026-001",
            target_cluster_id="NER-POT-01",
            cluster_description="Resurfaced Sector 4 road",
            region="Nerul",
            action_type="Road Resurfacing",
            department="PWD",
            created_timestamp="2026-03-01T10:00:00",
            status="Completed",
            notes="Completed ahead of schedule",
            is_simulated=False,
        )
        assert save_intervention_append(rec1, json_path=json_path) is True

        loaded = load_interventions(json_path=json_path)
        assert len(loaded) == 1
        assert loaded[0].intervention_id == "INT-2026-001"
        assert loaded[0].is_simulated is False
        assert loaded[0].region == "Nerul"


def test_compute_before_after_zero_baseline():
    """Verify safe handling of zero complaints prior to intervention."""
    interv = InterventionRecord(
        intervention_id="INT-TEST",
        target_cluster_id="TEST-01",
        cluster_description="Test",
        region="Nerul",
        action_type="Test",
        department="PWD",
        created_timestamp="2026-03-01T10:00:00",
    )
    # No complaints in window
    res = compute_before_after(interv, [], window_days=14)
    assert res["before_count"] == 0
    assert res["after_count"] == 0
    assert res["pct_change"] is None
    assert "N/A" in res["pct_label"]
    assert "Observational data only" in res["observation_disclaimer"]


def test_compute_before_after_with_complaints():
    """Verify complaint counting before and after split date."""
    interv = InterventionRecord(
        intervention_id="INT-TEST-2",
        target_cluster_id="TEST-02",
        cluster_description="Test",
        region="Nerul",
        action_type="Test",
        department="PWD",
        created_timestamp="2026-03-15T12:00:00",
    )

    c1 = EnrichedComplaint(
        complaint_id="CMP-1",
        region="Nerul",
        complaint_text="Pothole before intervention",
        submitted_at="2026-03-10T10:00:00",
        cleaned_text="pothole",
        detected_language="en",
        issue_category="Roads & Footpaths",
        resolved_location=ResolvedLocation(latitude=19.0, longitude=73.0),
    )
    c2 = EnrichedComplaint(
        complaint_id="CMP-2",
        region="Nerul",
        complaint_text="Another pothole before",
        submitted_at="2026-03-12T10:00:00",
        cleaned_text="pothole",
        detected_language="en",
        issue_category="Roads & Footpaths",
        resolved_location=ResolvedLocation(latitude=19.0, longitude=73.0),
    )
    c3 = EnrichedComplaint(
        complaint_id="CMP-3",
        region="Nerul",
        complaint_text="Single pothole after",
        submitted_at="2026-03-20T10:00:00",
        cleaned_text="pothole",
        detected_language="en",
        issue_category="Roads & Footpaths",
        resolved_location=ResolvedLocation(latitude=19.0, longitude=73.0),
    )

    res = compute_before_after(interv, [c1, c2, c3], window_days=14)
    assert res["before_count"] == 2
    assert res["after_count"] == 1
    assert res["pct_change"] == -50.0
    assert res["pct_label"] == "-50.0%"
    assert "Observational data only" in res["observation_disclaimer"]


def test_interpret_cluster_context_without_key():
    """Verify that interpret_cluster_context_with_llm returns None safely if no key is present."""
    res = interpret_cluster_context_with_llm(
        cluster_id="CLU-01",
        landmark_name="Station Road",
        region="Nerul",
        category="Roads & Footpaths",
        pattern_type="Recurring Infrastructure Issue",
        deterministic_score=75.0,
        priority_level="High",
        complaint_records=[],
    )
    # Empty complaint records or unconfigured environment must return None safely
    assert res is None


def test_pipeline_span_days():
    """Verify that _span_days correctly handles timestamps and returns float days."""
    pipeline = CivicPulsePipeline(force_fallback_embeddings=True)
    ts_list = ["2026-03-01T10:00:00", "2026-03-03T10:00:00"]
    days = pipeline._span_days(ts_list)
    assert round(days, 1) == 2.0


def test_incident_map_located_clusters():
    """Verify that pipeline clusters have valid coordinates for map rendering."""
    pipeline = CivicPulsePipeline(force_fallback_embeddings=True)
    pipeline.load_and_process_all()
    located = [c for c in pipeline.clusters if c.latitude is not None and c.longitude is not None]
    assert len(located) > 0
    for c in located:
        assert 18.0 <= c.latitude <= 20.0
        assert 72.0 <= c.longitude <= 74.0


