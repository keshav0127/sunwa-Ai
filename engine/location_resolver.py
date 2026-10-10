"""Sunwa-AI — Deterministic Location Resolver + Groq LLM Fallback.
Sequence:
  1. Unicode/whitespace normalization
  2. Exact alias matching (curated multilingual catalogue)
  3. Fuzzy matching (RapidFuzz, token-set ratio >= 80%)
  4. Region compatibility validation
  5. Ambiguity detection
  6. Groq LLM fallback only if deterministic fails + fallback enabled
  7. Strict catalogue validation of LLM suggestion
"""
import re
import unicodedata
import logging
from typing import List, Optional, Tuple, Dict
import pandas as pd
from rapidfuzz import fuzz

from models.schemas import ResolvedLocation, LocationStatus, LocationPrecision, MatchType

logger = logging.getLogger(__name__)

CONFIDENT_THRESHOLD = 0.80   # minimum confidence to call a match "resolved"
FUZZY_MIN_SCORE = 80         # RapidFuzz token-set ratio minimum (0-100)
AMBIGUOUS_MIN_COUNT = 2      # how many candidates trigger ambiguity


def _norm(text: str) -> str:
    """Normalize text for matching: NFC, strip, lowercase Latin, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase only Latin chars; keep Devanagari intact
    text = re.sub(r"[a-zA-Z]+", lambda m: m.group().lower(), text)
    return text


class LandmarkCatalogue:
    """Index of curated landmark entries for fast lookup."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy() if not df.empty else pd.DataFrame()
        self._index: Dict[str, Dict] = {}  # normalized_alias → landmark row dict
        self._build_index()

    def _build_index(self) -> None:
        if self._df.empty:
            return
        for _, row in self._df.iterrows():
            lm_name = str(row.get("landmark_name", "")).strip()
            aliases_raw = str(row.get("aliases", "")).strip()
            aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            all_names = [lm_name] + aliases
            record = row.to_dict()
            record["_all_names"] = all_names
            for name in all_names:
                key = _norm(name)
                if key:
                    self._index[key] = record

    def all_landmarks(self) -> List[Dict]:
        seen = set()
        rows = []
        for rec in self._index.values():
            lid = rec.get("landmark_id", "")
            if lid not in seen:
                seen.add(lid)
                rows.append(rec)
        return rows

    def get_candidates_for_region(self, region: str) -> List[Dict]:
        """All landmark records for a given region."""
        region_norm = region.strip().lower()
        return [rec for rec in self.all_landmarks()
                if rec.get("region", "").strip().lower() == region_norm]

    def get_landmarks_for_region(self, region: str) -> List[str]:
        """Distinct landmark names for a region (for LLM candidate list)."""
        return [rec["landmark_name"] for rec in self.get_candidates_for_region(region)]

    def lookup_name(self, name: str) -> Optional[Dict]:
        """Exact normalized lookup."""
        return self._index.get(_norm(name))

    def is_empty(self) -> bool:
        return len(self._index) == 0


class LocationResolver:
    """Deterministic location resolver using the curated landmark catalogue."""

    def __init__(self, catalogue: LandmarkCatalogue) -> None:
        self._cat = catalogue

    def resolve(self, complaint_text: str, region: str, location_hint: str = "") -> ResolvedLocation:
        """Run deterministic resolution pipeline."""
        combined = f"{complaint_text} {location_hint}".strip()
        norm_text = _norm(combined)

        region_candidates = self._cat.get_candidates_for_region(region)
        if not region_candidates:
            return ResolvedLocation(
                location_status=LocationStatus.UNRESOLVED.value,
                resolution_reason=f"No landmarks catalogued for region '{region}'",
            )

        # ── PHASE 1: Exact / alias substring matching ──────────────────
        exact_matches: List[Tuple[Dict, float]] = []
        for rec in region_candidates:
            for alias in rec.get("_all_names", []):
                alias_norm = _norm(alias)
                if alias_norm and alias_norm in norm_text:
                    score = min(1.0, len(alias_norm) / max(len(norm_text), 1) + 0.6)
                    exact_matches.append((rec, min(score, 1.0)))
                    break  # one alias match per landmark sufficient

        # Pick distinct top matches
        best_exact: Dict[str, Tuple[Dict, float]] = {}
        for rec, sc in exact_matches:
            lid = rec["landmark_id"]
            if lid not in best_exact or sc > best_exact[lid][1]:
                best_exact[lid] = (rec, sc)

        top_exact = sorted(best_exact.values(), key=lambda x: -x[1])

        if len(top_exact) == 1:
            return self._make_resolved(top_exact[0][0], top_exact[0][1],
                                       MatchType.ALIAS_MATCH.value, "Exact alias match in complaint text")
        if len(top_exact) >= AMBIGUOUS_MIN_COUNT:
            candidates = [r[0]["landmark_name"] for r in top_exact]
            return ResolvedLocation(
                location_status=LocationStatus.AMBIGUOUS.value,
                candidate_landmarks=candidates,
                match_confidence=0.45,
                resolution_reason=f"Ambiguous — multiple alias matches: {candidates}",
            )

        # ── PHASE 2: Fuzzy token-set matching ──────────────────────────
        fuzzy_matches: List[Tuple[Dict, float]] = []
        for rec in region_candidates:
            best_score = 0
            for alias in rec.get("_all_names", []):
                alias_norm = _norm(alias)
                if not alias_norm:
                    continue
                score = fuzz.token_set_ratio(norm_text, alias_norm)
                if score > best_score:
                    best_score = score
            if best_score >= FUZZY_MIN_SCORE:
                fuzzy_matches.append((rec, best_score / 100.0))

        fuzzy_matches.sort(key=lambda x: -x[1])

        if len(fuzzy_matches) == 1:
            return self._make_resolved(fuzzy_matches[0][0], fuzzy_matches[0][1],
                                       MatchType.FUZZY_MATCH.value, f"Fuzzy token-set ratio {fuzzy_matches[0][1]:.2f}")
        if len(fuzzy_matches) >= AMBIGUOUS_MIN_COUNT:
            candidates = [r[0]["landmark_name"] for r in fuzzy_matches[:3]]
            return ResolvedLocation(
                location_status=LocationStatus.AMBIGUOUS.value,
                candidate_landmarks=candidates,
                match_confidence=0.35,
                resolution_reason=f"Ambiguous — multiple fuzzy candidates: {candidates}",
            )

        # Unresolved
        return ResolvedLocation(
            location_status=LocationStatus.UNRESOLVED.value,
            resolution_reason="No landmark match found in complaint text or location hint",
        )

    def _make_resolved(self, rec: Dict, confidence: float, match_type: str, reason: str) -> ResolvedLocation:
        lat = rec.get("latitude")
        lng = rec.get("longitude")
        try:
            lat = float(lat) if lat is not None else None
            lng = float(lng) if lng is not None else None
        except (TypeError, ValueError):
            lat = lng = None
        return ResolvedLocation(
            matched_landmark=rec["landmark_name"],
            latitude=lat,
            longitude=lng,
            match_type=match_type,
            match_confidence=round(min(confidence, 1.0), 3),
            location_precision=LocationPrecision.LANDMARK.value,
            location_status=LocationStatus.RESOLVED.value,
            resolution_reason=reason,
        )

    def resolve_with_fallback(
        self,
        complaint_text: str,
        region: str,
        location_hint: str = "",
        llm_fallback_fn=None,
    ) -> ResolvedLocation:
        """Deterministic-first resolution. Invokes LLM fallback only if deterministic fails."""
        result = self.resolve(complaint_text, region, location_hint)

        if result.location_status == LocationStatus.RESOLVED.value:
            # Confident deterministic match — never call LLM
            return result

        # Eligible for LLM fallback
        if llm_fallback_fn is None:
            return result

        try:
            candidates = self._cat.get_landmarks_for_region(region)
            if not candidates:
                return result

            llm_suggestion = llm_fallback_fn(
                complaint_text=complaint_text,
                region=region,
                candidates=candidates,
            )

            if not llm_suggestion:
                return result

            # Validate suggestion strictly against catalogue
            rec = self._cat.lookup_name(llm_suggestion)
            if rec and rec.get("region", "").strip().lower() == region.strip().lower():
                lat = rec.get("latitude")
                lng = rec.get("longitude")
                try:
                    lat = float(lat) if lat is not None else None
                    lng = float(lng) if lng is not None else None
                except (TypeError, ValueError):
                    lat = lng = None
                return ResolvedLocation(
                    matched_landmark=rec["landmark_name"],
                    latitude=lat,
                    longitude=lng,
                    match_type="llm_assisted",
                    match_confidence=0.60,
                    location_precision=LocationPrecision.LANDMARK.value,
                    location_status=LocationStatus.RESOLVED.value,
                    resolution_reason=f"LLM-assisted match (unverified) → '{rec['landmark_name']}'. Requires human review.",
                )
            else:
                logger.warning("LLM suggested '%s' not found in catalogue for region '%s' — rejected.",
                               llm_suggestion, region)
                return result

        except Exception as exc:
            logger.error("LLM fallback error: %s — keeping deterministic result.", exc)
            return result
