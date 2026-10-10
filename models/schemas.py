"""Sunwa-AI — Canonical Data Schemas & Models.
Defines verified data structures for raw complaints, resolved locations,
enriched complaints, priority scoring, action recommendations, incident clusters,
actions, alerts, and interventions.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class MatchType(Enum):
    EXACT_LANDMARK = "exact_landmark"
    ALIAS_MATCH = "alias_match"
    FUZZY_MATCH = "fuzzy_match"
    SECTOR_MATCH = "sector_match"
    REGION_ONLY = "region_only"
    UNRESOLVED = "unresolved"


class LocationPrecision(Enum):
    LANDMARK = "landmark"
    SECTOR = "sector"
    REGION = "region"
    UNKNOWN = "unknown"


class LocationStatus(Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    REGION_ONLY = "region_only"
    UNRESOLVED = "unresolved"


class ActionStatus(Enum):
    PENDING_REVIEW = "pending_review"
    READY_TO_DISPATCH = "ready_to_dispatch"
    ALERT_QUEUED = "alert_queued"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class AlertStatus(Enum):
    SENT_SIMULATED = "sent_simulated"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class RawComplaint:
    """Original citizen grievance input record."""
    complaint_id: str
    region: str
    complaint_text: str
    submitted_at: str
    location_description: Optional[str] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complaint_id": self.complaint_id,
            "region": self.region,
            "complaint_text": self.complaint_text,
            "submitted_at": self.submitted_at,
            "location_description": self.location_description or "",
        }


@dataclass
class ResolvedLocation:
    """Geographic resolution output from deterministic resolver or LLM fallback."""
    matched_landmark: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    match_type: str = MatchType.UNRESOLVED.value
    match_confidence: float = 0.0
    location_precision: str = LocationPrecision.UNKNOWN.value
    location_status: str = LocationStatus.UNRESOLVED.value
    candidate_landmarks: List[str] = field(default_factory=list)
    resolution_reason: str = "No geographic reference resolved from text"

    @property
    def match_score(self) -> float:
        """Percentage match score (0-100) derived from match_confidence."""
        return round(self.match_confidence * 100.0 if self.match_confidence <= 1.0 else self.match_confidence, 1)

    @property
    def confidence(self) -> float:
        """Alias for match_confidence."""
        return self.match_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_landmark": self.matched_landmark,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "match_type": self.match_type,
            "match_confidence": self.match_confidence,
            "match_score": self.match_score,
            "location_precision": self.location_precision,
            "location_status": self.location_status,
            "candidate_landmarks": self.candidate_landmarks,
            "resolution_reason": self.resolution_reason,
        }


@dataclass
class EnrichedComplaint:
    """Complaint enriched with NLP classification, safety flags, and resolved location."""
    complaint_id: str
    region: str
    complaint_text: str
    submitted_at: str
    cleaned_text: str
    detected_language: str
    issue_category: str
    resolved_location: ResolvedLocation
    severity_indicators: List[str] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None
    cluster_assignment_method: str = "unassigned"
    image_path: Optional[str] = None
    image_status: str = "none"


@dataclass
class PriorityScore:
    """Transparent numerical priority breakdown P = 0.25S + 0.20A + 0.15D + 0.25R + 0.15T."""
    severity_score: float = 50.0       # S (0-100)
    affected_pop_score: float = 50.0   # A (0-100)
    duration_score: float = 50.0       # D (0-100)
    safety_risk_score: float = 0.0     # R (0-100)
    recurrence_score: float = 0.0      # T (0-100)
    overall_score: float = 40.0
    priority_level: str = "Medium"     # Critical (80-100), High (60-79), Medium (40-59), Low (0-39)
    safety_override_triggered: bool = False
    override_reason: str = ""
    explanation_breakdown: List[str] = field(default_factory=list)
    formula_expression: str = "P = 0.25*S + 0.20*A + 0.15*D + 0.25*R + 0.15*T"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity_score": self.severity_score,
            "affected_pop_score": self.affected_pop_score,
            "duration_score": self.duration_score,
            "safety_risk_score": self.safety_risk_score,
            "recurrence_score": self.recurrence_score,
            "overall_score": self.overall_score,
            "priority_level": self.priority_level,
            "safety_override_triggered": self.safety_override_triggered,
            "override_reason": self.override_reason,
            "formula_expression": self.formula_expression,
        }


@dataclass
class ActionRecommendation:
    """Deterministic or LLM-suggested administrative recommendation."""
    suggested_department: str
    recommended_action: str
    reasoning: str
    requires_human_review: bool = True
    missing_info: List[str] = field(default_factory=list)


@dataclass
class IncidentCluster:
    """Spatio-temporal & semantic cluster of citizen complaints representing an incident."""
    cluster_id: str
    region: str
    landmark_name: str
    category: str
    pattern_type: str                  # Localized Incident, Complaint Surge, Recurring Infrastructure Issue, etc.
    complaint_ids: List[str]
    complaint_count: int
    time_range: str
    priority: PriorityScore
    recommendation: ActionRecommendation
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ai_interpretation: Optional[str] = None


@dataclass
class ActionRecord:
    """Administrative action task lifecycle object."""
    action_id: str
    target_cluster_id: str
    category: str
    priority_level: str
    department: str
    action_type: str
    created_timestamp: str
    status: str = ActionStatus.PENDING_REVIEW.value
    notes: str = ""
    assigned_to: str = "Unassigned"
    acknowledged_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target_cluster_id": self.target_cluster_id,
            "category": self.category,
            "priority_level": self.priority_level,
            "department": self.department,
            "action_type": self.action_type,
            "created_timestamp": self.created_timestamp,
            "status": self.status,
            "notes": self.notes,
            "assigned_to": self.assigned_to,
            "acknowledged_at": self.acknowledged_at,
            "completed_at": self.completed_at,
        }


@dataclass
class AlertRecord:
    """Simulated municipal dispatch alert payload."""
    alert_id: str
    action_id: str
    cluster_id: str
    department: str
    priority_level: str
    message: str
    evidence_complaint_ids: List[str]
    created_at: str
    delivery_mode: str = "simulated"
    demo_label: str = "DEMO MODE — SIMULATED MUNICIPAL ALERT"
    alert_status: str = AlertStatus.SENT_SIMULATED.value
    acknowledged_at: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "action_id": self.action_id,
            "cluster_id": self.cluster_id,
            "department": self.department,
            "priority_level": self.priority_level,
            "message": self.message,
            "evidence_complaint_ids": self.evidence_complaint_ids,
            "created_at": self.created_at,
            "delivery_mode": self.delivery_mode,
            "demo_label": self.demo_label,
            "alert_status": self.alert_status,
            "acknowledged_at": self.acknowledged_at,
            "retry_count": self.retry_count,
        }


@dataclass
class InterventionRecord:
    """Administrative intervention record for before/after complaint activity analysis."""
    intervention_id: str
    target_cluster_id: str
    cluster_description: str
    region: str
    action_type: str
    department: str
    created_timestamp: str
    completion_timestamp: Optional[str] = None
    status: str = "In Progress"
    notes: str = ""
    is_simulated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "target_cluster_id": self.target_cluster_id,
            "cluster_description": self.cluster_description,
            "region": self.region,
            "action_type": self.action_type,
            "department": self.department,
            "created_timestamp": self.created_timestamp,
            "completion_timestamp": self.completion_timestamp,
            "status": self.status,
            "notes": self.notes,
            "is_simulated": self.is_simulated,
        }
