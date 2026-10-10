"""Sunwa-AI — Executive Overview Component."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List
from collections import Counter

from locales.i18n import t
from models.schemas import IncidentCluster, InterventionRecord


PRIORITY_COLORS = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#EAB308",
    "Low": "#22C55E",
}


def render_overview(
    clusters: List[IncidentCluster],
    complaints: list,
    interventions: List[InterventionRecord],
    selected_region: str,
) -> None:
    st.markdown(f"### 📊 {t('tab_overview')}")
    st.caption(t("overview_desc"))

    if not complaints:
        st.info(t("no_data"))
        return

    # ── KPI Metrics ──────────────────────────────────────────────────────
    total = len(complaints)
    n_clusters = len(clusters)
    critical_high = sum(1 for c in clusters if c.priority.priority_level in ("Critical", "High"))
    unlocated = sum(1 for c in complaints if c.resolved_location.location_status != "resolved")
    safety_hazards = sum(1 for c in clusters if c.pattern_type == "Potential Acute Hazard")
    n_interventions = len(interventions)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🗂️ Total Complaints", total)
    m2.metric("🔗 Incident Clusters", n_clusters)
    m3.metric("🚨 Critical/High", critical_high)
    m4.metric("📍 Unresolved Locations", unlocated)
    m5.metric("⚠️ Safety Hazards", safety_hazards)
    m6.metric("🔬 Interventions", n_interventions)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── Priority Distribution Pie ─────────────────────────────────────────
    with col_left:
        st.markdown("##### Priority Distribution")
        if clusters:
            counts = Counter(c.priority.priority_level for c in clusters)
            levels = ["Critical", "High", "Medium", "Low"]
            vals = [counts.get(l, 0) for l in levels]
            colors = [PRIORITY_COLORS[l] for l in levels]
            fig = go.Figure(data=[go.Pie(
                labels=levels,
                values=vals,
                marker_colors=colors,
                hole=0.5,
                textinfo="label+percent",
            )])
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0",
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No clusters to display.")

    # ── Category Bar ──────────────────────────────────────────────────────
    with col_right:
        st.markdown("##### Complaint Categories")
        if complaints:
            cat_counts = Counter(c.issue_category for c in complaints)
            df_cat = pd.DataFrame(cat_counts.items(), columns=["Category", "Count"]).sort_values("Count", ascending=True)
            fig2 = px.bar(
                df_cat, x="Count", y="Category", orientation="h",
                color_discrete_sequence=["#38BDF8"],
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0", height=280,
                xaxis_title="", yaxis_title="",
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Pattern Types ─────────────────────────────────────────────────────
    st.markdown("##### Incident Pattern Types")
    if clusters:
        pat_counts = Counter(c.pattern_type for c in clusters)
        df_pat = pd.DataFrame(pat_counts.items(), columns=["Pattern", "Count"]).sort_values("Count", ascending=False)
        st.dataframe(df_pat, use_container_width=True, hide_index=True)
    else:
        st.info("No pattern data.")

    # ── Complaints Timeline ────────────────────────────────────────────────
    st.markdown("##### Complaint Activity Timeline")
    ts_data = []
    for c in complaints:
        try:
            ts_data.append({"date": c.submitted_at[:10], "category": c.issue_category})
        except Exception:
            pass
    if ts_data:
        df_ts = pd.DataFrame(ts_data)
        df_ts_grouped = df_ts.groupby(["date", "category"]).size().reset_index(name="count")
        fig3 = px.area(
            df_ts_grouped, x="date", y="count", color="category",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E2E8F0", height=300,
            xaxis_title="Date", yaxis_title="Complaint Count",
            legend_title="Category",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig3, use_container_width=True)
