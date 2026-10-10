"""Tests: Data loading, schema, complaint model integrity."""
import os
import sys
import tempfile
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data_loader import load_complaints, load_landmarks, load_interventions, save_complaints_append
from models.schemas import RawComplaint, ResolvedLocation, LocationStatus, LocationPrecision, MatchType


# ── RawComplaint / Schema ────────────────────────────────────────────────────

def test_raw_complaint_schema():
    r = RawComplaint("CMP-001", "Nerul", "Pothole near SIES", "2026-09-20T08:00:00")
    assert r.complaint_id == "CMP-001"
    assert r.complaint_text == "Pothole near SIES"

def test_resolved_location_defaults():
    loc = ResolvedLocation()
    assert loc.location_status == LocationStatus.UNRESOLVED.value
    assert loc.match_confidence == 0.0
    assert loc.match_score == 0.0   # property
    assert loc.confidence == 0.0    # alias

def test_resolved_location_match_score_property():
    loc = ResolvedLocation(match_confidence=0.95)
    assert abs(loc.match_score - 95.0) < 0.1
    assert loc.confidence == 0.95

def test_resolved_location_to_dict():
    loc = ResolvedLocation(
        matched_landmark="SIES GST",
        latitude=19.0435,
        longitude=73.0234,
        match_confidence=0.92,
        location_status=LocationStatus.RESOLVED.value,
    )
    d = loc.to_dict()
    assert d["matched_landmark"] == "SIES GST"
    assert d["match_score"] == loc.match_score
    assert d["location_status"] == "resolved"


# ── CSV Loading ──────────────────────────────────────────────────────────────

def make_csv(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


def test_load_complaints_valid():
    path = make_csv(
        "complaint_id,region,complaint_text,submitted_at\n"
        "CMP-001,Nerul,Pothole near college,2026-09-20T08:00:00\n"
        "CMP-002,Panvel,Waterlogging at station,2026-09-21T09:00:00\n"
    )
    complaints = load_complaints(path)
    assert len(complaints) == 2
    assert complaints[0].complaint_id == "CMP-001"
    assert complaints[1].region == "Panvel"
    os.unlink(path)


def test_load_complaints_missing_optional_column():
    # No location_description column — should load fine
    path = make_csv(
        "complaint_id,region,complaint_text,submitted_at\n"
        "CMP-001,Nerul,Test complaint,2026-09-20T08:00:00\n"
    )
    complaints = load_complaints(path)
    assert len(complaints) == 1
    assert complaints[0].location_description is None
    os.unlink(path)


def test_load_complaints_empty_file():
    path = make_csv("complaint_id,region,complaint_text,submitted_at\n")
    complaints = load_complaints(path)
    assert complaints == []
    os.unlink(path)


def test_load_complaints_missing_required_column():
    path = make_csv("complaint_id,complaint_text,submitted_at\n")
    complaints = load_complaints(path)
    # missing 'region' — returns empty
    assert complaints == []
    os.unlink(path)


def test_load_complaints_malformed_row_skipped():
    path = make_csv(
        "complaint_id,region,complaint_text,submitted_at\n"
        ",Nerul,Empty ID here,2026-09-20T08:00:00\n"   # empty id
        "CMP-002,Panvel,Valid,2026-09-21T09:00:00\n"
    )
    complaints = load_complaints(path)
    assert len(complaints) == 1
    assert complaints[0].complaint_id == "CMP-002"
    os.unlink(path)


def test_load_complaints_duplicate_id_skipped():
    path = make_csv(
        "complaint_id,region,complaint_text,submitted_at\n"
        "CMP-001,Nerul,First,2026-09-20T08:00:00\n"
        "CMP-001,Nerul,Duplicate ID,2026-09-21T09:00:00\n"
    )
    complaints = load_complaints(path)
    assert len(complaints) == 1
    os.unlink(path)


def test_load_complaints_original_text_preserved():
    original = "एसआयईएस कॉलेजच्या मुख्य प्रवेशद्वाराजवळ खड्डा पडला आहे"
    path = make_csv(
        f"complaint_id,region,complaint_text,submitted_at\n"
        f'CMP-001,Nerul,"{original}",2026-09-20T08:00:00\n'
    )
    complaints = load_complaints(path)
    assert complaints[0].complaint_text == original
    os.unlink(path)


# ── Landmarks Loading ────────────────────────────────────────────────────────

def test_load_landmarks_actual_file():
    df = load_landmarks()
    if not df.empty:
        assert "landmark_id" in df.columns
        assert "region" in df.columns
        assert "latitude" in df.columns


def test_load_landmarks_missing_file():
    df = load_landmarks(Path("/nonexistent/landmarks.csv"))
    assert df.empty


# ── Append Complaints ────────────────────────────────────────────────────────

def test_save_complaints_append():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("complaint_id,region,complaint_text,submitted_at,location_description\n")
        f.write("CMP-001,Nerul,Existing complaint,2026-09-20T08:00:00,\n")
        path = Path(f.name)
    new_complaint = RawComplaint("CMP-NEW-001", "Panvel", "Test new complaint", "2026-09-25T10:00:00")
    result = save_complaints_append(new_complaint, path)
    assert result is True
    loaded = load_complaints(path)
    assert any(c.complaint_id == "CMP-NEW-001" for c in loaded)
    os.unlink(path)


def test_save_complaints_append_prevents_duplicate():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("complaint_id,region,complaint_text,submitted_at,location_description\n")
        f.write("CMP-001,Nerul,Existing,2026-09-20T08:00:00,\n")
        path = Path(f.name)
    dup = RawComplaint("CMP-001", "Nerul", "Duplicate", "2026-09-21T08:00:00")
    result = save_complaints_append(dup, path)
    assert result is False
    os.unlink(path)
