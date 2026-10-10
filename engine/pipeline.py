"""Sunwa-AI — Master Processing Pipeline.
Orchestrates: data load → normalize → locate → embed → cluster → prioritize → recommend.
Provides region/filter-aware accessors for dashboard components.
"""
import logging
import uuid
from datetime import datetime
from collections import defaultdict
from typing import List, Optional, Dict, Any

import numpy as np

from engine.data_loader import load_complaints, load_landmarks, load_interventions, save_complaints_append, save_intervention_append
from engine.complaint_normalizer import normalize_text, classify_category, extract_severity, extract_safety_flags, detect_language
from engine.location_resolver import LocationResolver, LandmarkCatalogue
from engine.llm_client import resolve_location_with_llm
from engine.semantic_engine import SemanticEngine
from engine.clustering_engine import cluster_complaints
from engine.pattern_analyzer import classify_pattern
from engine.priority_engine import compute_priority
from engine.recommendation_engine import get_recommendation
from models.schemas import (
    RawComplaint, EnrichedComplaint, ResolvedLocation, IncidentCluster,
    PriorityScore, LocationStatus,
)

logger = logging.getLogger(__name__)


class CivicPulsePipeline:
    """Main Sunwa-AI orchestration pipeline."""

    def __init__(self, force_fallback_embeddings: bool = False) -> None:
        self.force_fallback = force_fallback_embeddings
        self.embedder = SemanticEngine(force_fallback=force_fallback_embeddings)
        self.enriched_complaints: List[EnrichedComplaint] = []
        self.clusters: List[IncidentCluster] = []
        self.interventions = []
        self._raw_complaints: List[RawComplaint] = []
        self._catalogue: Optional[LandmarkCatalogue] = None
        self._resolver: Optional[LocationResolver] = None
        self._processed = False

    # ── Public loader ──────────────────────────────────────────────────────

    def load_and_process_all(self, refresh_cache: bool = False) -> None:
        if self._processed and not refresh_cache:
            return

        logger.info("Loading data...")
        raw_complaints = load_complaints()
        landmarks_df = load_landmarks()
        self.interventions = load_interventions()

        self._catalogue = LandmarkCatalogue(landmarks_df)
        self._resolver = LocationResolver(self._catalogue)

        logger.info("Normalizing and enriching %d complaints...", len(raw_complaints))
        enriched = []
        for raw in raw_complaints:
            ec = self._enrich(raw)
            enriched.append(ec)

        self._raw_complaints = raw_complaints
        self.enriched_complaints = enriched

        logger.info("Computing embeddings for %d complaints...", len(enriched))
        if enriched:
            texts = [c.cleaned_text for c in enriched]
            embeddings = self.embedder.embed(texts)
            self._cluster_and_prioritize(enriched, embeddings)
        else:
            self.clusters = []

        self._processed = True
        logger.info("Pipeline ready: %d complaints, %d clusters.", len(enriched), len(self.clusters))

    # ── Enrichment ────────────────────────────────────────────────────────

    def _enrich(self, raw: RawComplaint) -> EnrichedComplaint:
        cleaned = normalize_text(raw.complaint_text)
        lang = detect_language(raw.complaint_text)
        category = classify_category(raw.complaint_text)
        severity, sev_indicators = extract_severity(raw.complaint_text)
        safety_flags = extract_safety_flags(raw.complaint_text)

        # Location hint from location_description field
        loc_hint = raw.location_description or ""

        # Deterministic-first resolution, Groq fallback only if unresolved
        resolved = self._resolver.resolve_with_fallback(
            complaint_text=raw.complaint_text,
            region=raw.region,
            location_hint=loc_hint,
            llm_fallback_fn=resolve_location_with_llm,
        )

        return EnrichedComplaint(
            complaint_id=raw.complaint_id,
            region=raw.region,
            complaint_text=raw.complaint_text,
            submitted_at=raw.submitted_at,
            cleaned_text=cleaned,
            detected_language=lang,
            issue_category=category,
            resolved_location=resolved,
            severity_indicators=sev_indicators,
            safety_flags=safety_flags,
        )

    # ── Clustering & Prioritization ────────────────────────────────────────

    def _cluster_and_prioritize(
        self,
        enriched: List[EnrichedComplaint],
        embeddings: np.ndarray,
    ) -> None:
        cluster_map = cluster_complaints(enriched, embeddings)

        clusters: List[IncidentCluster] = []
        for clu_id, indices in cluster_map.items():
            members = [enriched[i] for i in indices]
            if not members:
                continue

            # Assign cluster
            for m in members:
                m.cluster_id = clu_id
                m.cluster_assignment_method = "semantic+geo+temporal"

            region = members[0].region
            category = self._majority_vote([m.issue_category for m in members])
            landmark = self._best_landmark(members)
            lat, lng = self._centroid(members)

            # Time range
            timestamps = [m.submitted_at for m in members]
            sorted_ts = sorted(timestamps)
            span_days = self._span_days(timestamps)
            time_range = f"{sorted_ts[0][:10]} → {sorted_ts[-1][:10]}" if sorted_ts else "Unknown"

            # Safety
            all_flags = []
            for m in members:
                all_flags.extend(m.safety_flags)
            all_flags = list(set(all_flags))

            severity = self._majority_vote([m.severity_indicators[0] if m.severity_indicators else "Low"
                                            for m in members])
            # Recalculate severity from members — use highest
            sev_priority = {"High": 3, "Medium": 2, "Low": 1}
            sev_levels = [extract_severity(m.complaint_text)[0] for m in members]
            severity = max(sev_levels, key=lambda s: sev_priority.get(s, 1))

            pattern = classify_pattern(
                complaint_count=len(members),
                region_count=len(set(m.region for m in members)),
                timestamps=timestamps,
                has_safety_flags=bool(all_flags),
                category=category,
                landmark_count=len(set(
                    m.resolved_location.matched_landmark for m in members
                    if m.resolved_location.matched_landmark
                )),
            )

            priority = compute_priority(
                severity=severity,
                complaint_count=len(members),
                span_days=span_days,
                safety_flags=all_flags,
                pattern_type=pattern,
            )

            recommendation = get_recommendation(
                category=category,
                priority_level=priority.priority_level,
                complaint_count=len(members),
                safety_flags=all_flags,
            )

            clusters.append(IncidentCluster(
                cluster_id=clu_id,
                region=region,
                landmark_name=landmark,
                category=category,
                pattern_type=pattern,
                complaint_ids=[m.complaint_id for m in members],
                complaint_count=len(members),
                time_range=time_range,
                priority=priority,
                recommendation=recommendation,
                latitude=lat,
                longitude=lng,
            ))

        # Sort by priority descending
        priority_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        clusters.sort(key=lambda c: priority_order.get(c.priority.priority_level, 0), reverse=True)
        self.clusters = clusters

    # ── Helpers ───────────────────────────────────────────────────────────

    def _majority_vote(self, items: List[str]) -> str:
        if not items:
            return "Unknown"
        from collections import Counter
        return Counter(items).most_common(1)[0][0]

    def _best_landmark(self, members: List[EnrichedComplaint]) -> str:
        for m in members:
            if m.resolved_location.matched_landmark:
                return m.resolved_location.matched_landmark
        return f"{members[0].region} — Unresolved Location"

    def _centroid(self, members: List[EnrichedComplaint]):
        lats = [m.resolved_location.latitude for m in members if m.resolved_location.latitude is not None]
        lngs = [m.resolved_location.longitude for m in members if m.resolved_location.longitude is not None]
        if lats and lngs:
            return sum(lats) / len(lats), sum(lngs) / len(lngs)
        return None, None

    def _span_days(self, timestamps: List[str]) -> float:
        from engine.clustering_engine import _parse_ts
        parsed = [_parse_ts(ts) for ts in timestamps if _parse_ts(ts) is not None]
        if not parsed:
            return 0.0
        return (max(parsed) - min(parsed)) / 86400.0

    # ── Accessors for dashboard ────────────────────────────────────────────

    def get_clusters_for_region(self, region: str) -> List[IncidentCluster]:
        if region == "All Supported Regions":
            return self.clusters
        return [c for c in self.clusters if c.region.strip().lower() == region.strip().lower()]

    def get_complaints_for_region(self, region: str) -> List[EnrichedComplaint]:
        if region == "All Supported Regions":
            return self.enriched_complaints
        return [c for c in self.enriched_complaints if c.region.strip().lower() == region.strip().lower()]

    def get_unlocated_complaints(self, region: str) -> List[EnrichedComplaint]:
        comps = self.get_complaints_for_region(region)
        return [c for c in comps if c.resolved_location.location_status != LocationStatus.RESOLVED.value]

    def get_unclustered_complaints(self, region: str) -> List[EnrichedComplaint]:
        comps = self.get_complaints_for_region(region)
        return [c for c in comps if c.cluster_id is None or c.cluster_assignment_method == "unassigned"]

    # ── Citizen portal: submit new complaint ────────────────────────────────

    def submit_complaint(self, region: str, text: str, location_hint: str = "") -> RawComplaint:
        """Create, enrich, append to CSV, and add to in-memory state."""
        now = datetime.utcnow().isoformat()
        cid = f"CMP-{datetime.utcnow().strftime('%Y')}-{uuid.uuid4().hex[:6].upper()}"
        raw = RawComplaint(
            complaint_id=cid,
            region=region,
            complaint_text=text,
            submitted_at=now,
            location_description=location_hint or None,
        )
        save_complaints_append(raw)
        ec = self._enrich(raw)
        self.enriched_complaints.append(ec)
        return raw

    def record_intervention(self, intervention) -> bool:
        """Add and persist a new administrative intervention."""
        self.interventions.append(intervention)
        return save_intervention_append(intervention)

