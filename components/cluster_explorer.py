"""Sunwa-AI — Semantic Cluster Explorer Component.
Displays semantic groupings, representative complaints, complaint counts,
time spans, geographic coordinates, and unclustered/outlier complaints.
Prevents similarity-chain confusion by visualizing cluster boundaries and constraints.
"""
import streamlit as st
import pandas as pd
from typing import List
from collections import Counter

from locales.i18n import t
from models.schemas import IncidentCluster, EnrichedComplaint, LocationStatus


def render_cluster_explorer(
    clusters: List[IncidentCluster],
    complaints: List[EnrichedComplaint],
    selected_region: str,
) -> None:
    st.markdown("### 🔗 Semantic Cluster Explorer")
    st.caption(
        "Incident clusters formed using multilingual semantic embeddings, geographic bounds, "
        "and temporal windows. Distinct physical locations and distant timeframes are preserved "
        "as separate incidents to prevent similarity-chain false merges."
    )

    if not clusters:
        st.info("No incident clusters available.")
        return

    # High level statistics
    total_in_clusters = sum(c.complaint_count for c in clusters)
    singletons = sum(1 for c in clusters if c.complaint_count == 1)
    multi_incident = len(clusters) - singletons
    avg_cohesion = f"{total_in_clusters / max(len(clusters), 1):.1f}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clusters", len(clusters))
    c2.metric("Multi-Complaint Incidents", multi_incident)
    c3.metric("Isolated / Outlier Incidents", singletons)
    c4.metric("Avg Complaints / Cluster", avg_cohesion)

    st.markdown("---")

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        cat_options = ["All Categories"] + sorted(list({c.category for c in clusters}))
        cat_filter = st.selectbox("Filter Category", cat_options, key="ce_cat")
    with f_col2:
        pattern_options = ["All Patterns"] + sorted(list({c.pattern_type for c in clusters}))
        pat_filter = st.selectbox("Filter Pattern Type", pattern_options, key="ce_pat")
    with f_col3:
        min_complaints = st.slider("Minimum Complaints", 1, max([c.complaint_count for c in clusters] or [1]), 1, key="ce_min")

    filtered = [
        c for c in clusters
        if (cat_filter == "All Categories" or c.category == cat_filter)
        and (pat_filter == "All Patterns" or c.pattern_type == pat_filter)
        and c.complaint_count >= min_complaints
    ]

    st.markdown(f"##### Showing {len(filtered)} Incident Clusters")

    # Map of complaint id to complaint object
    comp_map = {c.complaint_id: c for c in complaints}

    for cluster in filtered:
        p = cluster.priority
        priority_color = {
            "Critical": "#EF4444",
            "High": "#F97316",
            "Medium": "#EAB308",
            "Low": "#22C55E",
        }.get(p.priority_level, "#60A5FA")

        with st.expander(
            f"**{cluster.cluster_id}** · {cluster.landmark_name} [{cluster.region}] — "
            f"{cluster.complaint_count} Complaints · Priority: {p.priority_level} ({p.overall_score:.0f})",
            expanded=False,
        ):
            meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
            meta_c1.markdown(f"**Region:** {cluster.region}")
            meta_c2.markdown(f"**Category:** {cluster.category}")
            meta_c3.markdown(f"**Pattern:** `{cluster.pattern_type}`")
            meta_c4.markdown(f"**Time Window:** {cluster.time_range}")

            if cluster.latitude and cluster.longitude:
                st.caption(
                    f"📍 Centroid Coordinates: `{cluster.latitude:.4f}, {cluster.longitude:.4f}` "
                    f"(Verified landmark: *{cluster.landmark_name}*)"
                )
            else:
                st.caption("📍 Coordinates: *Unresolved / Approximate Landmark Boundary*")

            # Representative complaint & members
            cluster_members = [comp_map[cid] for cid in cluster.complaint_ids if cid in comp_map]

            if cluster_members:
                st.markdown("###### 📋 Cluster Members & Grounded Text")
                # Group languages in this cluster
                langs = Counter(m.detected_language for m in cluster_members)
                lang_str = ", ".join(f"{k.upper()}: {v}" for k, v in langs.items())
                st.caption(f"🌐 Multilingual composition: {lang_str}")

                for member in cluster_members[:10]:
                    loc_badge = "✅ Resolved" if member.resolved_location.location_status == "resolved" else "⚠️ Approximate"
                    st.markdown(
                        f"""
                        <div style="padding: 8px 12px; margin-bottom: 6px; border-radius: 6px;
                            background: rgba(255,255,255,0.03); border-left: 3px solid {priority_color};">
                            <span style="font-size: 0.8rem; color: #94A3B8;">
                                <b>{member.complaint_id}</b> · {member.submitted_at[:16]} · Lang: {member.detected_language} · Loc: {loc_badge}
                            </span><br/>
                            <span style="color: #F1F5F9; font-size: 0.95rem;">{member.complaint_text}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if len(cluster_members) > 10:
                    st.caption(f"+ {len(cluster_members) - 10} additional complaints in this cluster")

    # Outliers / Unclustered Section
    st.markdown("---")
    st.markdown("#### 🔍 Outliers & Unclustered Complaints")
    st.caption(
        "Complaints that did not meet semantic or geographic grouping thresholds are preserved as "
        "independent records to ensure no citizen grievance is discarded or hidden."
    )

    outliers = [
        c for c in complaints
        if c.cluster_assignment_method == "unassigned" or c.cluster_id is None
        or (c.cluster_id and any(clu.cluster_id == c.cluster_id and clu.complaint_count == 1 for clu in clusters))
    ]

    with st.expander(f"View Preserved Outlier & Isolated Complaints ({len(outliers)})"):
        for o in outliers[:25]:
            st.markdown(
                f"• **{o.complaint_id}** [{o.region} - {o.issue_category}]: \"{o.complaint_text}\" "
                f"*(Submitted: {o.submitted_at[:10]} | Status: {o.resolved_location.location_status})*"
            )
