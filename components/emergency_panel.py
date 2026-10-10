"""Sunwa-AI — Emergency Hazard Detection Panel.
Identifies urgent civic safety risks using deterministic keyword extraction,
safety overrides, and cluster pattern classification.
Explains specific triggering evidence. Does not claim physical verification.
Fully functional without LLM.
"""
import streamlit as st
from typing import List

from locales.i18n import t
from models.schemas import IncidentCluster, EnrichedComplaint


def render_emergency_panel(
    clusters: List[IncidentCluster],
    complaints: List[EnrichedComplaint],
) -> None:
    st.markdown("### ⚠️ Emergency Hazard Detection")
    st.caption(
        "Deterministic detection of urgent public safety risks. "
        "Safety overrides escalate issues directly to protect public welfare, "
        "preventing ordinary weighted formulas from diluting critical hazards. "
        "All alerts require on-ground human municipal verification."
    )

    comp_map = {c.complaint_id: c for c in complaints}

    # Find acute hazards or clusters with safety overrides
    hazard_clusters = [
        c for c in clusters
        if c.pattern_type == "Potential Acute Hazard"
        or c.priority.safety_override_triggered
        or c.priority.safety_risk_score >= 50
    ]

    # Individual complaints with safety flags
    flagged_complaints = [
        c for c in complaints
        if c.safety_flags
    ]

    h1, h2, h3 = st.columns(3)
    h1.metric("🚨 High Safety Risk Clusters", len(hazard_clusters))
    h2.metric("🚩 Total Flagged Complaints", len(flagged_complaints))
    h3.metric("⚡ Active Safety Overrides", sum(1 for c in clusters if c.priority.safety_override_triggered))

    st.markdown("---")

    if not hazard_clusters:
        st.success("✅ No acute public safety hazards currently flagged in this view.")
    else:
        st.markdown("#### 🚨 Priority Safety Incidents Requiring Human Inspection")
        for clu in hazard_clusters:
            p = clu.priority
            with st.container():
                st.markdown(
                    f"""
                    <div style="border: 2px solid #EF4444; border-radius: 8px; padding: 16px;
                        background: rgba(239, 68, 68, 0.05); margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.2rem; font-weight: 700; color: #FCA5A5;">
                                ⚠️ {clu.landmark_name} [{clu.region}]
                            </span>
                            <span style="background: #EF4444; color: white; padding: 4px 10px;
                                border-radius: 12px; font-size: 0.85rem; font-weight: 600;">
                                {p.priority_level.upper()} (Score: {p.overall_score})
                            </span>
                        </div>
                        <p style="margin: 8px 0 4px; color: #CBD5E1;">
                            <b>Cluster ID:</b> <code>{clu.cluster_id}</code> |
                            <b>Category:</b> {clu.category} |
                            <b>Complaints:</b> {clu.complaint_count} |
                            <b>Pattern:</b> <code>{clu.pattern_type}</code>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if p.safety_override_triggered:
                    st.error(f"🛡️ **Safety Escalation Triggered:** {p.override_reason}")

                st.markdown("##### 🔍 Evidence & Triggering Factors")
                exp_c1, exp_c2 = st.columns(2)
                with exp_c1:
                    st.markdown("**Deterministic Safety Signals:**")
                    # Gather safety flags from members
                    members = [comp_map[cid] for cid in clu.complaint_ids if cid in comp_map]
                    all_flags = set()
                    for m in members:
                        all_flags.update(m.safety_flags)
                    if all_flags:
                        for flag in all_flags:
                            st.markdown(f"- 🔴 Trigger keyword / flag: `\"{flag}\"`")
                    else:
                        st.markdown(f"- ⚠️ Escalated by pattern: `{clu.pattern_type}`")

                with exp_c2:
                    st.markdown("**Immediate Recommended Protocol:**")
                    st.info(
                        f"**Department:** {clu.recommendation.suggested_department}\n\n"
                        f"**Action:** {clu.recommendation.recommended_action}\n\n"
                        f"*Verification Note: Ground truth must be physically inspected by emergency crews.*"
                    )

                with st.expander(f"Inspect Original Supporting Excerpts ({len(members)})"):
                    for m in members:
                        st.markdown(
                            f"• **[{m.complaint_id}]** *({m.detected_language})*: \"{m.complaint_text}\""
                        )
                st.markdown("---")

    # Flagged individual grievances
    st.markdown("#### 🚩 All Grievances With Potential Safety Indicators")
    st.caption("Individual citizen complaints matching emergency keywords (e.g. fire, open wire, collapse, collision, accident, shock).")

    with st.expander(f"View All {len(flagged_complaints)} Grievances with Safety Flags"):
        for fc in flagged_complaints[:30]:
            st.markdown(
                f"- **{fc.complaint_id}** [{fc.region}]: \"{fc.complaint_text}\"  \n"
                f"  *Flags:* `{', '.join(fc.safety_flags)}` | *Severity Proxy:* `{', '.join(fc.severity_indicators) or 'Standard'}`"
            )
