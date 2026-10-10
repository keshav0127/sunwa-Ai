"""Tests: Deterministic location resolver — exact, fuzzy, Devanagari, multilingual, ambiguous, unresolved."""
import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.location_resolver import LandmarkCatalogue, LocationResolver
from models.schemas import LocationStatus, MatchType


def make_catalogue() -> LandmarkCatalogue:
    rows = [
        {
            "landmark_id": "LM-NER-001",
            "region": "Nerul",
            "landmark_name": "SIES Graduate School of Technology",
            "aliases": "SIES College|SIES GST|SIES Engineering College|एसआयईएस|एसआयईएस कॉलेज|एसआयईएस जीएसटी",
            "category": "Education",
            "latitude": "19.0435",
            "longitude": "73.0234",
            "coordinate_source": "OSM",
            "verification_status": "verified",
        },
        {
            "landmark_id": "LM-NER-002",
            "region": "Nerul",
            "landmark_name": "Nerul Railway Station",
            "aliases": "Nerul Station|Nerul Stn|नेरुल स्टेशन|नेरुळ स्टेशन|नेरुल रेल्वे स्टेशन",
            "category": "Transit",
            "latitude": "19.0330",
            "longitude": "73.0182",
            "coordinate_source": "OSM",
            "verification_status": "verified",
        },
        {
            "landmark_id": "LM-PAN-001",
            "region": "Panvel",
            "landmark_name": "Panvel Railway Station",
            "aliases": "Panvel Station|Panvel Stn|पनवेल स्टेशन|पनवेल रेल्वे स्टेशन",
            "category": "Transit",
            "latitude": "18.9902",
            "longitude": "73.1205",
            "coordinate_source": "OSM",
            "verification_status": "verified",
        },
    ]
    df = pd.DataFrame(rows)
    return LandmarkCatalogue(df)


@pytest.fixture
def cat():
    return make_catalogue()


@pytest.fixture
def resolver(cat):
    return LocationResolver(cat)


# ── Exact / Alias Matching ───────────────────────────────────────────────────

def test_exact_alias_english(resolver):
    result = resolver.resolve("Huge pothole near SIES College entrance", "Nerul")
    assert result.location_status == LocationStatus.RESOLVED.value
    assert "SIES" in result.matched_landmark
    assert result.latitude is not None


def test_exact_alias_abbreviation(resolver):
    result = resolver.resolve("Road issue near SIES GST gate", "Nerul")
    assert result.location_status == LocationStatus.RESOLVED.value


def test_devanagari_alias_matching(resolver):
    result = resolver.resolve("एसआयईएस कॉलेज जवळ मोठा खड्डा आहे", "Nerul")
    assert result.location_status == LocationStatus.RESOLVED.value
    assert "SIES" in result.matched_landmark


def test_devanagari_station_alias(resolver):
    result = resolver.resolve("नेरुल स्टेशन बाहेर पाणी साचले आहे", "Nerul")
    assert result.location_status == LocationStatus.RESOLVED.value
    assert "Nerul Railway Station" == result.matched_landmark


def test_romanized_hindi_matching(resolver):
    result = resolver.resolve("Nerul station ke bahar khadda hai bada", "Nerul")
    # Fuzzy should pick up Nerul Station
    assert result.location_status in (LocationStatus.RESOLVED.value, LocationStatus.AMBIGUOUS.value)


def test_panvel_station_alias(resolver):
    result = resolver.resolve("पनवेल स्टेशन बाहेर पाणी साचले आहे गटार तुंबले", "Panvel")
    assert result.location_status == LocationStatus.RESOLVED.value
    assert "Panvel Railway Station" == result.matched_landmark


# ── Region Compatibility ─────────────────────────────────────────────────────

def test_region_mismatch_unresolved(resolver):
    # "SIES College" is in Nerul, not Panvel
    result = resolver.resolve("SIES College ke pass khamba gira", "Panvel")
    assert result.location_status == LocationStatus.UNRESOLVED.value


def test_correct_region_resolves(resolver):
    result = resolver.resolve("Panvel Station ke bahar kachra pada hai", "Panvel")
    assert result.location_status in (LocationStatus.RESOLVED.value, LocationStatus.AMBIGUOUS.value)


# ── Ambiguous / Unresolved ───────────────────────────────────────────────────

def test_unresolved_no_match(resolver):
    result = resolver.resolve("Some vague place in the municipality", "Nerul")
    assert result.location_status in (LocationStatus.UNRESOLVED.value, LocationStatus.AMBIGUOUS.value)


def test_empty_catalogue_unresolved():
    empty_cat = LandmarkCatalogue(pd.DataFrame())
    resolver = LocationResolver(empty_cat)
    result = resolver.resolve("SIES College pothole", "Nerul")
    assert result.location_status == LocationStatus.UNRESOLVED.value


# ── Coordinates Only From Catalogue ─────────────────────────────────────────

def test_coordinates_from_catalogue(resolver):
    result = resolver.resolve("SIES GST gate broken road", "Nerul")
    if result.location_status == LocationStatus.RESOLVED.value:
        assert result.latitude == pytest.approx(19.0435, abs=0.001)
        assert result.longitude == pytest.approx(73.0234, abs=0.001)


def test_no_invented_coordinates(resolver):
    result = resolver.resolve("Some random location nowhere", "Nerul")
    if result.location_status == LocationStatus.UNRESOLVED.value:
        assert result.latitude is None
        assert result.longitude is None


# ── LLM Fallback Integration ─────────────────────────────────────────────────

def test_llm_fallback_not_called_for_resolved(resolver):
    """Confident deterministic match must never call LLM."""
    called = []

    def fake_llm(**kwargs):
        called.append(True)
        return "SIES Graduate School of Technology"

    result = resolver.resolve_with_fallback(
        "SIES College ke samne bada gaddha hai",
        "Nerul",
        llm_fallback_fn=fake_llm,
    )
    assert result.location_status == LocationStatus.RESOLVED.value
    assert len(called) == 0, "LLM should NOT have been called for a resolved deterministic match"


def test_llm_fallback_called_for_unresolved(resolver):
    """Unresolved deterministic results should invoke LLM fallback."""
    called = []

    def fake_llm(complaint_text, region, candidates, **kwargs):
        called.append(True)
        return "SIES Graduate School of Technology"

    result = resolver.resolve_with_fallback(
        "office wala kaam nahi chal raha kuch",
        "Nerul",
        llm_fallback_fn=fake_llm,
    )
    # Fallback should have been called
    assert len(called) == 1


def test_invented_landmark_rejected(resolver):
    """LLM-suggested landmark not in catalogue must be rejected."""
    def fake_llm(complaint_text, region, candidates, **kwargs):
        return "Non-Existent Invented Landmark XYZ"

    result = resolver.resolve_with_fallback(
        "some random text with no location clues nearby",
        "Nerul",
        llm_fallback_fn=fake_llm,
    )
    # Should NOT resolve to invented landmark
    assert result.matched_landmark != "Non-Existent Invented Landmark XYZ"


def test_llm_none_fallback_returns_unresolved(resolver):
    result = resolver.resolve_with_fallback(
        "some vague text with no location",
        "Nerul",
        llm_fallback_fn=None,
    )
    assert result.location_status in (LocationStatus.UNRESOLVED.value, LocationStatus.AMBIGUOUS.value)
