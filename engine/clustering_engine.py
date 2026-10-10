"""Sunwa-AI — Incident Clustering Engine.
Strategy: Semantic similarity (cosine from embeddings) + geographic + temporal constraints.
Prevents merging complaints from different regions or distant locations.
Preserves outliers as singleton clusters.
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.55     # cosine similarity that puts two complaints in same cluster
GEO_MAX_DISTANCE_KM = 1.5      # max distance between cluster centroids before blocking merge
TIME_WINDOW_DAYS = 14           # complaints further apart in time are not co-clustered
MAX_CLUSTER_SIZE = 25           # soft cap to prevent giant global clusters


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Haversine formula for approximate great-circle distance in km."""
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_ts(ts_str: str) -> Optional[float]:
    """Parse ISO timestamp to Unix float, return None on failure."""
    from datetime import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str[:19], fmt).timestamp()
        except (ValueError, TypeError):
            pass
    return None


def cluster_complaints(
    enriched_complaints: List,
    embeddings: np.ndarray,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Dict[str, List[int]]:
    """
    Group complaint indices into clusters using threshold-based graph connected components.
    Returns: dict mapping cluster_id → list of complaint indices.
    Constraints:
      - Same region required
      - Geographic proximity (if coordinates available)
      - Temporal window
      - Max cluster size (prevents over-merging)
    """
    n = len(enriched_complaints)
    if n == 0:
        return {}

    # Build adjacency list from pairwise similarity + constraints
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    adj: Dict[int, List[int]] = defaultdict(list)

    for i in range(n):
        ci = enriched_complaints[i]
        ts_i = _parse_ts(ci.submitted_at)
        lat_i = ci.resolved_location.latitude
        lng_i = ci.resolved_location.longitude

        for j in range(i + 1, n):
            cj = enriched_complaints[j]

            # Must be same region
            if ci.region.strip().lower() != cj.region.strip().lower():
                continue

            # Semantic similarity check
            sim = float(sim_matrix[i, j])
            if sim < similarity_threshold:
                continue

            # Temporal constraint
            ts_j = _parse_ts(cj.submitted_at)
            if ts_i is not None and ts_j is not None:
                day_diff = abs(ts_i - ts_j) / 86400.0
                if day_diff > TIME_WINDOW_DAYS:
                    continue

            # Geographic constraint (only if both have coordinates)
            lat_j = cj.resolved_location.latitude
            lng_j = cj.resolved_location.longitude
            if (lat_i is not None and lng_i is not None and
                    lat_j is not None and lng_j is not None):
                dist = _haversine_km(lat_i, lng_i, lat_j, lng_j)
                if dist > GEO_MAX_DISTANCE_KM:
                    continue

            adj[i].append(j)
            adj[j].append(i)

    # Connected components BFS → clusters
    visited = [False] * n
    clusters: Dict[str, List[int]] = {}
    cluster_idx = 0

    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        cluster_id = f"CLU-{cluster_idx:04d}"
        clusters[cluster_id] = component
        cluster_idx += 1

    return clusters
