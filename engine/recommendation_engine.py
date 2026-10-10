"""Sunwa-AI — Deterministic Recommendation Engine.
Maps issue categories → responsible departments and recommended actions.
Uses rule-based logic, not LLM.
"""
from typing import List, Optional
from models.schemas import ActionRecommendation

# Department and action rule tables
_RULES = [
    {
        "category": "Road Damage & Potholes",
        "department": "Roads & Public Works",
        "actions": [
            "Dispatch inspection team to assess pothole extent and road surface condition.",
            "Arrange emergency hot-mix asphalt patching with compaction verification.",
            "Erect road hazard warning signage pending repair.",
        ],
    },
    {
        "category": "Drainage & Flooding",
        "department": "Drainage & Sanitation",
        "actions": [
            "Deploy high-pressure jetting unit to clear storm drain blockage.",
            "Inspect drain capacity and desilting requirements.",
            "Set temporary drainage diversion pumps if flooding is ongoing.",
        ],
    },
    {
        "category": "Solid Waste & Sanitation",
        "department": "Solid Waste Management",
        "actions": [
            "Deploy compactor vehicle for bulk waste clearance.",
            "Increase bin installation at identified hotspot.",
            "Schedule daily collection cycle for the location.",
        ],
    },
    {
        "category": "Water Supply & Pipeline",
        "department": "Water Supply Department",
        "actions": [
            "Dispatch plumber team to identify and isolate leakage point.",
            "Issue water supply advisory to affected zone.",
            "Arrange repair work after isolation of supply line section.",
        ],
    },
    {
        "category": "Electrical & Streetlights",
        "department": "Electrical Maintenance",
        "actions": [
            "Send line crew to inspect faulty streetlight circuit.",
            "Replace blown lamps/controllers within 24 hours per safety protocol.",
            "Verify night-time illumination after repair.",
        ],
    },
    {
        "category": "Tree / Vegetation",
        "department": "Parks & Horticulture",
        "actions": [
            "Dispatch tree-trimming crew to remove obstruction or fallen branches.",
            "Assess tree stability and arrange removal if necessary.",
        ],
    },
    {
        "category": "Footpath & Pavement",
        "department": "Roads & Public Works",
        "actions": [
            "Inspect footpath damage extent.",
            "Arrange paving repair or tile replacement work.",
        ],
    },
    {
        "category": "Sewage & Manholes",
        "department": "Drainage & Sanitation",
        "actions": [
            "Inspect open manhole and secure with cover immediately.",
            "Assess sewage overflow source and arrange suction cleaning.",
        ],
    },
]

_DEFAULT_RULE = {
    "category": "General Infrastructure",
    "department": "Public Works — Manual Review",
    "actions": [
        "Forward complaint to relevant department for initial inspection.",
        "Record complaint and request site verification report.",
    ],
}


def get_recommendation(
    category: str,
    priority_level: str,
    complaint_count: int,
    safety_flags: Optional[List[str]] = None,
    missing_info: Optional[List[str]] = None,
) -> ActionRecommendation:
    """Deterministic rule-based department recommendation."""
    rule = next((r for r in _RULES if r["category"] == category), _DEFAULT_RULE)

    # Select action based on priority
    actions = rule["actions"]
    if priority_level in ("Critical", "High") and len(actions) >= 2:
        action = "; ".join(actions[:2])
    else:
        action = actions[0]

    missing = missing_info or []
    reason_parts = [f"Issue category '{category}' maps to {rule['department']}."]
    if safety_flags:
        reason_parts.append(f"Safety flags detected: {safety_flags}.")
    if priority_level in ("Critical", "High"):
        reason_parts.append(f"Priority '{priority_level}' — multiple actions triggered.")
    if complaint_count >= 5:
        reason_parts.append(f"{complaint_count} related complaints indicate systemic issue.")

    return ActionRecommendation(
        suggested_department=rule["department"],
        recommended_action=action,
        reasoning=" ".join(reason_parts),
        requires_human_review=True,
        missing_info=missing,
    )
