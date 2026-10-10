"""Sunwa-AI — Intervention Tracking Component.
Allows administrators to record municipal interventions and evaluate
before/after complaint activity across customizable time windows.
All results are clearly designated observational only (correlation != causation).
Persists intervention records across application restarts.
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from typing import List

from locales.i18n import t
from models.schemas import InterventionRecord, IncidentCluster, EnrichedComplaint
from engine.intervention_manager import compute_before_after


def render_intervention_view(
    interventions: List[InterventionRecord],
    clusters: List[IncidentCluster],
    complaints: List[EnrichedComplaint],
    pipeline=None,
) -> None:
    st.markdown("### 🔬 Intervention Tracking & Observational Analysis")
    st.caption(
        "Record municipal maintenance and administrative actions, then evaluate "
        "complaint volume before and after the intervention date. "
        "Results reflect observational trends; they do not establish strict physical causation."
    )
    st.warning(
        "⚠️ **Observational Integrity Notice:** "
        "A decrease in public complaints following an intervention does not prove "
        "the intervention caused the reduction. Confounding variables (weather, reporting fatigue, "
        "unrelated seasonal changes) may exist."
    )

    tab_eval, tab_create = st.tabs(["📊 Evaluation & Trends", "📝 Record New Intervention"])

    # ── TAB 1: Evaluation ────────────────────────────────────────────────
    with tab_eval:
        if not interventions:
            st.info("No intervention records logged yet. Use the 'Record New Intervention' tab to create one.")
        else:
            # Window days selector
            window_days = st.slider("Evaluation Window (Days Before & After)", min_value=3, max_value=60, value=14, step=1)

            st.markdown(f"##### Tracking {len(interventions)} Municipal Interventions")

            for interv in reversed(interventions):
                sim_badge = "🔬 Simulated Demo" if interv.is_simulated else "🏛️ Official Record"
                with st.expander(
                    f"**{interv.intervention_id}** · {interv.region} · {interv.action_type} · "
                    f"[{interv.status}] · {sim_badge}",
                    expanded=False,
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Department:** {interv.department}")
                    c2.markdown(f"**Created Date:** `{interv.created_timestamp[:10]}`")
                    c3.markdown(f"**Completed Date:** `{interv.completion_timestamp[:10] if interv.completion_timestamp else 'In Progress'}`")

                    st.markdown(f"**Description:** {interv.cluster_description}")
                    if interv.notes:
                        st.caption(f"**Field Notes:** {interv.notes}")

                    # Linked cluster
                    linked_cluster = next((c for c in clusters if c.cluster_id == interv.target_cluster_id), None)
                    if linked_cluster:
                        st.caption(f"🔗 Linked Incident: **{linked_cluster.landmark_name}** ({linked_cluster.complaint_count} current reports)")

                    # Before / After Computation
                    res = compute_before_after(interv, complaints, window_days=window_days)

                    if "error" in res:
                        st.warning(res["error"])
                    else:
                        before = res["before_count"]
                        after = res["after_count"]

                        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                        col_kpi1.metric(f"Before ({window_days}d)", before)
                        col_kpi2.metric(f"After ({window_days}d)", after)

                        if res["pct_change"] is not None:
                            pct_val = res["pct_change"]
                            delta_color = "normal" if pct_val <= 0 else "inverse"
                            col_kpi3.metric("Change", res["pct_label"], delta=f"{pct_val}%", delta_color=delta_color)
                        else:
                            col_kpi3.metric("Change", res["pct_label"])

                        # Plotly Before/After comparison bar
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[f"Before Intervention ({window_days}d)", f"After Intervention ({window_days}d)"],
                                y=[before, after],
                                marker_color=["#F97316", "#22C55E" if after <= before else "#EF4444"],
                                text=[before, after],
                                textposition="auto",
                            )
                        ])
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#E2E8F0",
                            height=200,
                            margin=dict(t=10, b=10, l=10, r=10),
                            yaxis_title="Complaint Count",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Supporting complaint IDs
                        with st.expander("View Complaining Records in Observation Window"):
                            st.caption(f"• Pre-intervention Complaint IDs ({len(res['before_complaint_ids'])}): {', '.join(res['before_complaint_ids']) or 'None'}")
                            st.caption(f"• Post-intervention Complaint IDs ({len(res['after_complaint_ids'])}): {', '.join(res['after_complaint_ids']) or 'None'}")

    # ── TAB 2: Record New Intervention ───────────────────────────────────
    with tab_create:
        st.markdown("#### 📝 Record New Administrative Intervention")
        st.caption("Logs an intervention against an incident cluster or pattern. Persisted across application restarts.")

        with st.form("new_intervention_form", clear_on_submit=False):
            # Cluster selector
            cluster_choices = ["General / Region-wide"] + [
                f"{c.cluster_id} — {c.landmark_name} ({c.region}) [{c.priority.priority_level}]"
                for c in clusters
            ]
            selected_choice = st.selectbox("Target Incident Cluster", cluster_choices)

            col_form1, col_form2 = st.columns(2)
            with col_form1:
                action_type = st.selectbox(
                    "Intervention Action Type",
                    [
                        "Road Resurfacing & Pothole Patching",
                        "Drainage Unblocking & Desilting",
                        "Water Pipeline Repair & Valve Replacement",
                        "Streetlight Repair & Wiring Overhaul",
                        "Garbage Clearance & Sanitation Drive",
                        "Traffic Signal Calibration & Signage",
                        "Vector Control / Fumigation",
                        "Structural Inspection & Barricading",
                    ]
                )
                department = st.selectbox(
                    "Responsible Department",
                    [
                        "Public Works Department (PWD)",
                        "Water Supply & Sewerage",
                        "Solid Waste Management",
                        "Electrical / Street Lighting",
                        "Traffic Management",
                        "Health & Sanitation",
                        "Disaster Management Cell",
                    ]
                )
                interv_region = st.selectbox(
                    "Region",
                    ["Nerul", "Panvel", "Vashi", "Kharghar", "Belapur"],
                )
            with col_form2:
                interv_date = st.date_input("Intervention Date", value=datetime.utcnow().date())
                status = st.selectbox("Status", ["In Progress", "Completed", "Scheduled", "Monitoring"])
                is_simulated = st.checkbox("Mark as Demonstration / Simulated", value=False)

            description = st.text_input(
                "Intervention Description",
                placeholder="e.g. Deployed quick-setting asphalt crew to patch potholes along main corridor.",
            )
            notes = st.text_area(
                "Field Notes & Observations",
                placeholder="e.g. Work carried out between 22:00 and 04:00 to avoid traffic congestion.",
                height=80,
            )

            submit_interv = st.form_submit_button("💾 Save Intervention Record", type="primary")

        if submit_interv:
            if not description.strip():
                st.error("Please enter a description for the intervention.")
            else:
                target_id = "GENERAL"
                if selected_choice != "General / Region-wide":
                    target_id = selected_choice.split(" — ")[0]

                new_id = f"INT-{datetime.utcnow().strftime('%Y%m%d')}-{len(interventions)+1:03d}"
                created_iso = f"{interv_date.isoformat()}T10:00:00"

                new_record = InterventionRecord(
                    intervention_id=new_id,
                    target_cluster_id=target_id,
                    cluster_description=description.strip(),
                    region=interv_region,
                    action_type=action_type,
                    department=department,
                    created_timestamp=created_iso,
                    completion_timestamp=created_iso if status == "Completed" else None,
                    status=status,
                    notes=notes.strip(),
                    is_simulated=is_simulated,
                )

                if pipeline:
                    pipeline.record_intervention(new_record)
                else:
                    interventions.append(new_record)

                st.success(f"✅ Intervention **{new_id}** saved and persisted successfully!")
                st.rerun()
