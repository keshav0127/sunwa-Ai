"""Sunwa-AI — Incident Map & Density Heatmap Component.
Visualizes incident clusters with priority markers and complaint-density heatmap.
Includes filters for priority, category, and view mode.
Never fabricates coordinates for unresolved locations; displays them in a separate panel.
"""
import streamlit as st
import pandas as pd
from typing import List, Optional

from locales.i18n import t
from models.schemas import IncidentCluster, EnrichedComplaint, LocationStatus

PRIORITY_COLOR = {
    "Critical": "red",
    "High": "orange",
    "Medium": "beige",
    "Low": "green",
}

PRIORITY_HEX = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#EAB308",
    "Low": "#22C55E",
}


def _render_folium_map(
    located_clusters: List[IncidentCluster],
    map_mode: str,
    complaint_records: List[EnrichedComplaint],
) -> bool:
    """Render Folium map with Markers, HeatMap, or Combined mode."""
    try:
        import folium
        from folium.plugins import HeatMap
        from streamlit_folium import st_folium

        if not located_clusters:
            return False

        center_lat = sum(c.latitude for c in located_clusters) / len(located_clusters)
        center_lng = sum(c.longitude for c in located_clusters) / len(located_clusters)

        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=13,
            tiles="OpenStreetMap",
        )

        # 1. Density HeatMap layer
        if map_mode in ("Complaint Density Heatmap", "Combined (Markers + Heatmap)"):
            heat_data = [
                [c.latitude, c.longitude, float(c.complaint_count)]
                for c in located_clusters
            ]
            HeatMap(
                heat_data,
                radius=25,
                blur=18,
                max_zoom=14,
                gradient={0.2: "#38BDF8", 0.5: "#F59E0B", 0.8: "#EF4444", 1.0: "#991B1B"},
            ).add_to(m)

        # 2. Incident Markers layer
        if map_mode in ("Incident Markers", "Combined (Markers + Heatmap)"):
            comp_map = {c.complaint_id: c.complaint_text for c in complaint_records}

            for cluster in located_clusters:
                color = PRIORITY_COLOR.get(cluster.priority.priority_level, "blue")
                first_text = "No excerpts available."
                for cid in cluster.complaint_ids:
                    if cid in comp_map:
                        first_text = comp_map[cid][:120] + ("..." if len(comp_map[cid]) > 120 else "")
                        break

                popup_html = (
                    f"<div style='font-family: sans-serif; font-size: 12px; width: 240px;'>"
                    f"<h4 style='margin:0 0 4px 0; color:#1E293B;'>{cluster.landmark_name}</h4>"
                    f"<span style='background:{PRIORITY_HEX.get(cluster.priority.priority_level, '#333')}; "
                    f"color:white; padding:2px 6px; border-radius:4px; font-weight:bold;'>"
                    f"{cluster.priority.priority_level} ({cluster.priority.overall_score:.0f})</span><br/><br/>"
                    f"<b>Complaints:</b> {cluster.complaint_count}<br/>"
                    f"<b>Category:</b> {cluster.category}<br/>"
                    f"<b>Pattern:</b> {cluster.pattern_type}<br/>"
                    f"<b>Department:</b> {cluster.recommendation.suggested_department}<br/>"
                    f"<hr style='margin: 6px 0; border: none; border-top: 1px solid #CBD5E1;'/>"
                    f"<i>\"{first_text}\"</i><br/>"
                    f"<small style='color:#64748B;'>Cluster: {cluster.cluster_id}</small>"
                    f"</div>"
                )
                folium.CircleMarker(
                    location=[cluster.latitude, cluster.longitude],
                    radius=8 + min(cluster.complaint_count * 2, 16),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.75,
                    popup=folium.Popup(popup_html, max_width=280),
                    tooltip=f"{cluster.landmark_name} · {cluster.priority.priority_level} ({cluster.complaint_count} reports)",
                ).add_to(m)

        st_folium(
            m,
            use_container_width=True,
            height=520,
            key="incident_map_folium_view",
            returned_objects=[],
        )
        return True
    except Exception as exc:
        st.warning(f"Folium interactive map notice ({exc}). Rendering native map view.")
        return False


def _render_native_map(located_clusters: List[IncidentCluster]) -> None:
    """Native Streamlit interactive map fallback."""
    try:
        map_data = []
        for c in located_clusters:
            map_data.append({
                "lat": c.latitude,
                "lon": c.longitude,
                "Landmark": c.landmark_name,
                "Priority": c.priority.priority_level,
                "Score": round(c.priority.overall_score, 1),
                "Complaints": c.complaint_count,
                "Category": c.category,
                "Pattern": c.pattern_type,
            })
        df_map = pd.DataFrame(map_data)
        st.map(df_map, latitude="lat", longitude="lon", size=15, color="#38BDF8")
    except Exception as exc:
        st.error(f"Native map fallback error: {exc}")


def render_hotspot_map(
    clusters: List[IncidentCluster],
    unlocated_complaints: list,
    all_complaints: Optional[List[EnrichedComplaint]] = None,
) -> None:
    st.markdown("### 🗺️ Incident Map & Complaint Density Hotmap")
    st.caption(
        "Geographic distribution of resolved civic incidents. "
        "Coordinates are derived from verified landmark centroids in the curated municipal catalogue. "
        "Unresolved complaints are strictly excluded from coordinates to prevent false localization."
    )

    if not clusters:
        st.info("No incident clusters available to map.")
        return

    # In-page Filter & View Controls
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        map_mode = st.radio(
            "Map Display Layer",
            ["Combined (Markers + Heatmap)", "Incident Markers", "Complaint Density Heatmap"],
            horizontal=False,
            key="map_layer_selection_radio",
        )
    with col_v2:
        selected_priorities = st.multiselect(
            "Filter Priority",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
            key="map_priority_multiselect",
        )
    with col_v3:
        all_cats = sorted(list({c.category for c in clusters}))
        selected_cats = st.multiselect(
            "Filter Category",
            all_cats,
            default=all_cats,
            key="map_category_multiselect",
        )

    # Filtered clusters
    filtered_clusters = [
        c for c in clusters
        if c.priority.priority_level in selected_priorities
        and c.category in selected_cats
    ]

    located_clusters = [c for c in filtered_clusters if c.latitude is not None and c.longitude is not None]
    unlocated_clusters = [c for c in filtered_clusters if c.latitude is None or c.longitude is None]

    # Legend
    legend_cols = st.columns(4)
    for i, (level, color) in enumerate(PRIORITY_HEX.items()):
        legend_cols[i].markdown(
            f'<span style="color:{color}; font-size:1.2em;">●</span> <b>{level}</b>',
            unsafe_allow_html=True,
        )

    # Render Map
    if located_clusters:
        complaint_records = all_complaints or []
        folium_success = _render_folium_map(located_clusters, map_mode, complaint_records)
        if not folium_success:
            _render_native_map(located_clusters)
    else:
        st.info("No located incidents match the active filters.")

    # ── Unresolved Locations Section ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📍 Unresolved & Ambiguous Location Grievances")
    st.caption(
        "The system never fabricates coordinates or assumes arbitrary positions. "
        "Complaints with ambiguous, missing, or unverified locations remain in the priority queue "
        "and are preserved here for manual municipal review."
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown(f"**Unlocated Incident Clusters:** {len(unlocated_clusters)}")
        for uc in unlocated_clusters[:10]:
            st.warning(
                f"**{uc.cluster_id}** · {uc.landmark_name} [{uc.region}] · "
                f"Priority: {uc.priority.priority_level} · {uc.complaint_count} complaints"
            )
    with col_u2:
        st.markdown(f"**Individual Unresolved Complaints:** {len(unlocated_complaints)}")
        with st.expander(f"View Unresolved Grievances ({len(unlocated_complaints)})"):
            for c in unlocated_complaints[:20]:
                st.caption(
                    f"• **{c.complaint_id}** [{c.region}]: \"{c.complaint_text[:110]}...\"  \n"
                    f"  *Status:* `{c.resolved_location.location_status}` | "
                    f"*Reason:* {c.resolved_location.resolution_reason}"
                )
