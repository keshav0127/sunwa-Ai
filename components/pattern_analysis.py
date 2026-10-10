"""Sunwa-AI — Recurrence and Systemic Pattern Analysis Component.
Distinguishes between:
  1. Localized incidents
  2. Short-term complaint surges
  3. Recurring infrastructure problems
  4. Similar problems across multiple locations (Distributed systemic issues)
  5. Potential emergency hazards
Grounds pattern hypotheses in temporal and spatial data without asserting
unverified physical root causes.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List
from collections import Counter

from locales.i18n import t
from models.schemas import IncidentCluster, EnrichedComplaint


def render_pattern_analysis(
    clusters: List[IncidentCluster],
    complaints: List[EnrichedComplaint],
) -> None:
    st.markdown("### 🔄 Recurrence & Systemic Pattern Analysis")
    st.caption(
        "Differentiating between isolated incidents, rapid complaint surges, "
        "chronic recurring infrastructure decay, and city-wide systemic problems. "
        "Observations indicate statistical patterns and operational hypotheses, "
        "not verified on-ground physical root causes."
    )

    if not clusters:
        st.info("No cluster data available for pattern analysis.")
        return

    # Count pattern occurrences
    pattern_counts = Counter(c.pattern_type for c in clusters)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Acute Hazards", pattern_counts.get("Potential Acute Hazard", 0))
    c2.metric("📈 Complaint Surges", pattern_counts.get("Complaint Surge", 0))
    c3.metric("🔄 Recurring Issues", pattern_counts.get("Recurring Infrastructure Issue", 0))
    c4.metric("🌐 Distributed Systemic", pattern_counts.get("Distributed Systemic Issue", 0))

    st.markdown("---")

    # Classification breakdown explanations
    with st.expander("ℹ️ How Sunwa-AI Distinguishes Patterns", expanded=False):
        st.markdown(
            """
            | Pattern Type | Defining Criteria | Operational Meaning |
            | :--- | :--- | :--- |
            | **Potential Acute Hazard** | Safety keywords detected (fire, collapse, open wire, collision) + multiple reports | Immediate life/safety hazard. Requires emergency triage. |
            | **Complaint Surge** | $\ge 4$ complaints within $\le 3$ days at same landmark | Rapid onset incident (e.g. sudden water main burst, traffic signal failure). |
            | **Recurring Infrastructure Issue** | Persistent complaints spanning $> 21$ days at same landmark | Chronic deterioration or failed previous repairs (e.g. repeated pothole reopenings). |
            | **Distributed Systemic Issue** | Similar grievance category distributed across $\ge 3$ landmarks or regions | Broad supply or vendor failure (e.g. regional power grid fluctuation, citywide garbage strike). |
            | **Localized Incident** | Standard localized issue with standard duration | Regular municipal grievance. Normal routing. |
            """
        )

    # Pattern selector filter
    pat_types = ["All"] + list(pattern_counts.keys())
    selected_pat = st.selectbox("Filter by Pattern Category", pat_types)

    display_clusters = clusters if selected_pat == "All" else [c for c in clusters if c.pattern_type == selected_pat]

    st.markdown(f"##### Analyzing {len(display_clusters)} Patterns")

    for clu in display_clusters:
        p = clu.priority
        with st.container():
            st.markdown(
                f"""
                <div style="border-left: 4px solid #38BDF8; padding: 12px 16px; border-radius: 8px;
                    background: rgba(255,255,255,0.03); margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <h4 style="margin: 0; color: #F8FAFC;">
                            {clu.landmark_name} · <span style="color: #38BDF8;">{clu.pattern_type}</span>
                        </h4>
                        <span style="color: #94A3B8; font-size: 0.9rem;">
                            Score: <b>{p.overall_score}</b> ({p.priority_level})
                        </span>
                    </div>
                    <p style="margin: 6px 0; color: #94A3B8; font-size: 0.9rem;">
                        Region: <b>{clu.region}</b> | Category: <b>{clu.category}</b> |
                        Time Window: <b>{clu.time_range}</b> | Complaints: <b>{clu.complaint_count}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Hypothesis and grounding rationale
            if clu.pattern_type == "Recurring Infrastructure Issue":
                st.warning(
                    f"🔄 **Recurrence Hypothesis:** Grievances for *{clu.category}* at *{clu.landmark_name}* "
                    f"have persisted across {clu.time_range} ({clu.complaint_count} reports). "
                    "This suggests repeated failure or insufficient permanent repair. "
                    "*Note: Physical inspection needed to verify underlying drainage/structural cause.*"
                )
            elif clu.pattern_type == "Complaint Surge":
                st.error(
                    f"📈 **Surge Hypothesis:** Sudden burst of {clu.complaint_count} grievances in a tight time window "
                    f"({clu.time_range}). Indicates acute service breakdown or sudden disruption."
                )
            elif clu.pattern_type == "Distributed Systemic Issue":
                st.info(
                    f"🌐 **Systemic Hypothesis:** Related complaints observed across multiple zones. "
                    "Treat as potential macro-level coordination issue rather than isolated local failure."
                )
            elif clu.pattern_type == "Potential Acute Hazard":
                st.error(
                    f"⚠️ **Hazard Flag:** Safety flags present in reports for *{clu.landmark_name}*. "
                    "Escalated priority to prevent life-safety hazards."
                )

            # Member complaints preview
            member_comps = [c for c in complaints if c.complaint_id in clu.complaint_ids]
            with st.expander(f"View Supporting Grievance Records ({len(member_comps)})"):
                for mc in member_comps:
                    st.caption(
                        f"• **{mc.complaint_id}** ({mc.submitted_at[:10]}): \"{mc.complaint_text}\""
                    )

    st.markdown("---")

    # Category by Pattern Matrix
    st.markdown("##### 📊 Pattern vs Category Matrix")
    matrix_data = []
    for c in clusters:
        matrix_data.append({"Pattern": c.pattern_type, "Category": c.category})
    if matrix_data:
        df_mat = pd.DataFrame(matrix_data)
        ct = pd.crosstab(df_mat["Category"], df_mat["Pattern"])
        st.dataframe(ct, use_container_width=True)
