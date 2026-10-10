"""Tests: Clustering engine — semantic grouping, geographic constraints, temporal constraints."""
import sys
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.clustering_engine import cluster_complaints
from models.schemas import EnrichedComplaint, ResolvedLocation, LocationStatus


def make_complaint(cid, region, text, lat=None, lng=None, ts=None):
    loc = ResolvedLocation(
        matched_landmark="SIES College" if lat else None,
        latitude=lat,
        longitude=lng,
        location_status=LocationStatus.RESOLVED.value if lat else LocationStatus.UNRESOLVED.value,
    )
    return EnrichedComplaint(
        complaint_id=cid,
        region=region,
        complaint_text=text,
        submitted_at=ts or "2026-09-20T10:00:00",
        cleaned_text=text.lower(),
        detected_language="en",
        issue_category="Road Damage & Potholes",
        resolved_location=loc,
    )


def _embed(texts):
    """Simple deterministic fake embeddings: same text → same vector."""
    rng = np.random.RandomState(42)
    vectors = []
    for t in texts:
        seed = abs(hash(t)) % (2**32)
        local_rng = np.random.RandomState(seed)
        vectors.append(local_rng.randn(64))
    mat = np.array(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def test_same_text_same_cluster():
    text = "pothole on the main road near college"
    c1 = make_complaint("C1", "Nerul", text, lat=19.0435, lng=73.0234)
    c2 = make_complaint("C2", "Nerul", text, lat=19.0435, lng=73.0234)
    texts = [c1.cleaned_text, c2.cleaned_text]
    embeddings = _embed(texts)
    # Force high cosine similarity by using identical embeddings
    embeddings = np.array([embeddings[0], embeddings[0]])
    clusters = cluster_complaints([c1, c2], embeddings, similarity_threshold=0.50)
    # Both should end up in same cluster
    all_indices = [idx for indices in clusters.values() for idx in indices]
    assert 0 in all_indices and 1 in all_indices
    cluster_with_both = [k for k, v in clusters.items() if 0 in v and 1 in v]
    assert len(cluster_with_both) == 1


def test_different_regions_not_clustered():
    c1 = make_complaint("C1", "Nerul", "pothole near college", lat=19.0435, lng=73.0234)
    c2 = make_complaint("C2", "Panvel", "pothole near college", lat=19.0435, lng=73.0234)
    texts = [c1.cleaned_text, c2.cleaned_text]
    embeddings = _embed(texts)
    # Force identical embeddings (same text) but different regions
    embeddings[1] = embeddings[0]
    clusters = cluster_complaints([c1, c2], embeddings)
    # Should be in separate clusters due to region mismatch
    assert len(clusters) >= 1
    cluster_with_both = [k for k, v in clusters.items() if 0 in v and 1 in v]
    assert len(cluster_with_both) == 0


def test_outliers_remain_visible():
    c1 = make_complaint("C1", "Nerul", "pothole near SIES", lat=19.0435, lng=73.0234)
    c2 = make_complaint("C2", "Nerul", "garbage collection issue completely different", lat=19.0330, lng=73.0182)
    texts = [c1.cleaned_text, c2.cleaned_text]
    embeddings = _embed(texts)
    clusters = cluster_complaints([c1, c2], embeddings, similarity_threshold=0.99)
    # Both should appear in some cluster
    all_indices = {idx for indices in clusters.values() for idx in indices}
    assert 0 in all_indices and 1 in all_indices


def test_empty_complaints():
    clusters = cluster_complaints([], np.zeros((0, 64)))
    assert clusters == {}


def test_single_complaint():
    c = make_complaint("C1", "Nerul", "pothole near college")
    embeddings = _embed([c.cleaned_text])
    clusters = cluster_complaints([c], embeddings)
    assert len(clusters) == 1
    all_indices = list(clusters.values())[0]
    assert 0 in all_indices


def test_temporal_separation_prevents_clustering():
    """Complaints far apart in time should not cluster."""
    c1 = make_complaint("C1", "Nerul", "pothole near college", lat=19.0435, lng=73.0234,
                        ts="2026-01-01T10:00:00")
    c2 = make_complaint("C2", "Nerul", "pothole near college", lat=19.0435, lng=73.0234,
                        ts="2026-09-01T10:00:00")  # 8 months later
    texts = [c1.cleaned_text, c2.cleaned_text]
    embeddings = _embed(texts)
    embeddings[1] = embeddings[0]  # force same embedding
    clusters = cluster_complaints([c1, c2], embeddings)
    cluster_with_both = [k for k, v in clusters.items() if 0 in v and 1 in v]
    assert len(cluster_with_both) == 0, "Temporally distant complaints should not cluster"
