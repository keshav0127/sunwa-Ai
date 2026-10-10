"""Sunwa-AI — Clean UI Components Package.
Action Manager and its workflows have been completely removed.
"""
from components.overview import render_overview
from components.incident_map import render_hotspot_map
from components.cluster_explorer import render_cluster_explorer
from components.priority_queue import render_priority_queue
from components.emergency_panel import render_emergency_panel
from components.evidence_panel import render_evidence_panel
from components.pattern_analysis import render_pattern_analysis
from components.intervention_tracking import render_intervention_view
from components.citizen_portal import render_citizen_portal

__all__ = [
    "render_overview",
    "render_hotspot_map",
    "render_cluster_explorer",
    "render_priority_queue",
    "render_emergency_panel",
    "render_evidence_panel",
    "render_pattern_analysis",
    "render_intervention_view",
    "render_citizen_portal",
]
