"""Sunwa-AI — Priority Queue Component.
Displays deterministic numerical priority scoring (formula P = 0.25S + 0.20A + 0.15D + 0.25R + 0.15T)
with safety escalation overrides.
Supports optional Groq LLM contextual interpretation grounded in actual complaint IDs,
strictly separated from the deterministic score.
"""
import streamlit as st
import pandas as pd
from typing import List

from locales.i18n import t
from models.schemas import IncidentCluster, EnrichedComplaint
from engine.llm_client import interpret_cluster_context_with_llm

PRIORITY_BADGE = {
    "Critical": "🔴 Critical",
    "High": "🟠 High",
    "Medium": "🟡 Medium",
    "Low": "🟢 Low",
}
PATTERN_ICON = {
    "Potential Acute Hazard": "⚠️",
    "Complaint Surge": "📈",
    "Recurring Infrastructure Issue": "🔄",
    "Distributed Systemic Issue": "🌐",
    "Localized Incident": "📍",
    "Insufficient Evidence": "❓",
}


def render_priority_queue(
    clusters: List[IncidentCluster],
    complaints: List[EnrichedComplaint],
) -> None:
    st.markdown(f"### 🚨 {t('tab_priority_queue')}")
    st.caption(
        "Incident clusters prioritized via deterministic, transparent formula: "
        "P = 0.25·S + 0.20·A + 0.15·D + 0.25·R + 0.15·T. "
        "LLMs never calculate or override numerical priority scores."
    )

    if not clusters:
        st.info(t("no_data"))
        return

    comp_map = {c.complaint_id: c for c in complaints}

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
        )
    with col_f2:
        all_patterns = list({c.pattern_type for c in clusters})
        pattern_filter = st.multiselect(
            "Filter by Pattern",
            all_patterns,
            default=all_patterns,
        )

    filtered = [
        c for c in clusters
        if c.priority.priority_level in priority_filter
        and c.pattern_type in pattern_filter
    ]

    st.caption(f"Showing {len(filtered)} of {len(clusters)} prioritized incident clusters")

    for cluster in filtered:
        p = cluster.priority
        icon = PATTERN_ICON.get(cluster.pattern_type, "📌")
        badge = PRIORITY_BADGE.get(p.priority_level, p.priority_level)

        with st.expander(
            f"{icon} **{cluster.landmark_name}** · {badge} · Score: {p.overall_score}/100 · {cluster.complaint_count} reports",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Deterministic Score", f"{p.overall_score}/100")
            c2.metric("Reports", cluster.complaint_count)
            c3.metric("Category", cluster.category)
            c4.metric("Pattern", cluster.pattern_type)

            st.caption(f"**Region:** {cluster.region} | **Window:** {cluster.time_range} | **Cluster ID:** `{cluster.cluster_id}`")

            if p.safety_override_triggered:
                st.error(f"🛡️ **Safety Escalation Override:** {p.override_reason}")

            # Score breakdown
            with st.expander("📊 Deterministic Score Formula & Factor Breakdown", expanded=False):
                st.code(p.formula_expression, language=None)
                cols = st.columns(5)
                cols[0].metric("Severity (S)", f"{p.severity_score:.0f}")
                cols[1].metric("Population (A)", f"{p.affected_pop_score:.0f}")
                cols[2].metric("Duration (D)", f"{p.duration_score:.0f}")
                cols[3].metric("Safety (R)", f"{p.safety_risk_score:.0f}")
                cols[4].metric("Recurrence (T)", f"{p.recurrence_score:.0f}")
                for line in p.explanation_breakdown:
                    st.caption(f"• {line}")

            # Operational recommendation
            rec = cluster.recommendation
            st.info(
                f"**🏢 Suggested Department:** {rec.suggested_department}  \n"
                f"**✅ Recommended Action:** {rec.recommended_action}  \n"
                f"**📝 Grounded Rationale:** {rec.reasoning}"
            )

            # Optional Groq Interpretation Section
            st.markdown("---")
            st.markdown("##### 🤖 Optional AI Contextual Interpretation (Groq)")
            st.caption("Generates a qualitative brief citing only verified complaint IDs. Does NOT alter the deterministic score.")

            cluster_member_records = [
                {
                    "complaint_id": cid,
                    "text": comp_map[cid].complaint_text,
                    "submitted_at": comp_map[cid].submitted_at,
                }
                for cid in cluster.complaint_ids if cid in comp_map
            ]

            if cluster.ai_interpretation:
                st.markdown(
                    f"""
                    <div style="border-left: 3px solid #818CF8; padding: 10px 14px; border-radius: 6px;
                        background: rgba(129, 140, 248, 0.05); margin-bottom: 10px;">
                        <span style="color: #A5B4FC; font-weight: 600; font-size: 0.85rem;">
                            🤖 AI-GENERATED CONTEXTUAL BRIEF (GROQ — GROUNDED IN EVIDENCE)
                        </span>
                        <div style="margin-top: 8px; color: #E2E8F0; font-size: 0.95rem; white-space: pre-wrap;">
{cluster.ai_interpretation}
                        </div>
                        <small style="color: #94A3B8; display: block; margin-top: 6px;">
                            *Official Score remains: {p.overall_score}/100 ({p.priority_level}). Grounded in {len(cluster_member_records)} complaints.*
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                interp_btn = st.button(
                    f"🤖 Request Groq Contextual Analysis ({cluster.cluster_id})",
                    key=f"interp_btn_{cluster.cluster_id}",
                )
                if interp_btn:
                    with st.spinner("Requesting grounded contextual interpretation from Groq..."):
                        ai_res = interpret_cluster_context_with_llm(
                            cluster_id=cluster.cluster_id,
                            landmark_name=cluster.landmark_name,
                            region=cluster.region,
                            category=cluster.category,
                            pattern_type=cluster.pattern_type,
                            deterministic_score=p.overall_score,
                            priority_level=p.priority_level,
                            complaint_records=cluster_member_records,
                        )
                        if ai_res:
                            cluster.ai_interpretation = ai_res
                            st.success("Contextual interpretation generated.")
                            st.rerun()
                        else:
                            st.warning(
                                "Groq interpretation unavailable (check API key or status). "
                                "Deterministic priority system operates independently and is fully intact."
                            )

            # Supporting Complaint IDs
            st.caption(
                f"**Supporting Complaint IDs ({len(cluster.complaint_ids)}):** "
                f"{', '.join(cluster.complaint_ids[:8])}" +
                (f" + {len(cluster.complaint_ids)-8} more" if len(cluster.complaint_ids) > 8 else "")
            )
