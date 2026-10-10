"""Sunwa-AI — Streamlit Application Entry Point.
Multilingual civic grievance intelligence platform.
Dual portal (Admin / Citizen), deterministic priority scoring, semantic clustering,
location resolution with Groq fallback, and observational intervention tracking.
Action Manager and its workflows have been completely removed.
"""
import sys
import logging
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables early (e.g., GROQ_API)
from dotenv import load_dotenv
load_dotenv()

# Engine and localization imports
from engine.pipeline import CivicPulsePipeline
from locales.i18n import t, set_current_lang, get_current_lang, SUPPORTED_LANGS
from components import (
    render_overview,
    render_hotspot_map,
    render_cluster_explorer,
    render_priority_queue,
    render_emergency_panel,
    render_evidence_panel,
    render_pattern_analysis,
    render_intervention_view,
    render_citizen_portal,
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline Initialization (Cached singleton)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline(force_fallback_embeddings: bool = False):
    pipeline = CivicPulsePipeline(force_fallback_embeddings=force_fallback_embeddings)
    pipeline.load_and_process_all()
    return pipeline

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom High-Contrast Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sunwa-AI (सुनवाई-AI)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-contrast CSS for sidebar and premium civic-tech styling
st.markdown(
    """
    <style>
        /* High contrast sidebar typography and background */
        section[data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid #1E293B !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div {
            color: #F8FAFC !important;
        }
        section[data-testid="stSidebar"] label {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            border-color: #334155 !important;
        }
        /* Metric badges in sidebar */
        .sidebar-stat-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }
        .sidebar-stat-label {
            font-size: 0.8rem;
            color: #94A3B8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .sidebar-stat-val {
            font-size: 1.3rem;
            font-weight: 700;
            color: #38BDF8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding-bottom: 12px; border-bottom: 1px solid #334155; margin-bottom: 16px;">
        <h2 style="margin: 0; color: #38BDF8; font-weight: 800;">सुनवाई-AI</h2>
        <span style="font-size: 0.85rem; color: #94A3B8;">Civic Grievance Intelligence</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Interface Language Selector
st.sidebar.markdown("### 🌐 " + t("language_selector"))
selected_lang = st.sidebar.selectbox(
    "Interface Language",
    options=SUPPORTED_LANGS,
    index=SUPPORTED_LANGS.index(get_current_lang()),
    label_visibility="collapsed",
)
set_current_lang(selected_lang)

# 2. Region Filter
st.sidebar.markdown("### 📍 " + t("ward_filter"))
@st.cache_data(show_spinner=False)
def load_regions():
    from engine.data_loader import load_complaints
    complaints = load_complaints()
    return sorted({c.region for c in complaints})

region_options = ["All Supported Regions"] + load_regions()
selected_region = st.sidebar.selectbox(
    "Region Filter",
    options=region_options,
    label_visibility="collapsed",
)

# 3. Pipeline Pipeline Initialization & Reprocess
if st.sidebar.button(t("reprocess_pipeline"), use_container_width=True):
    get_pipeline.clear()
    pipeline = get_pipeline()
    st.sidebar.success("Pipeline refreshed.")
else:
    pipeline = get_pipeline()

# 4. High-Contrast Sidebar Statistics
filtered_complaints = pipeline.get_complaints_for_region(selected_region)
filtered_clusters = pipeline.get_clusters_for_region(selected_region)
unlocated_count = len(pipeline.get_unlocated_complaints(selected_region))
hazard_count = sum(1 for c in filtered_clusters if c.pattern_type == "Potential Acute Hazard" or c.priority.safety_override_triggered)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 " + t("ward_stats"))
st.sidebar.markdown(
    f"""
    <div class="sidebar-stat-card">
        <div class="sidebar-stat-label">{t('total_complaints')}</div>
        <div class="sidebar-stat-val">{len(filtered_complaints)}</div>
    </div>
    <div class="sidebar-stat-card">
        <div class="sidebar-stat-label">{t('active_clusters')}</div>
        <div class="sidebar-stat-val" style="color: #818CF8;">{len(filtered_clusters)}</div>
    </div>
    <div class="sidebar-stat-card">
        <div class="sidebar-stat-label">Safety Hazards / Alerts</div>
        <div class="sidebar-stat-val" style="color: #EF4444;">{hazard_count}</div>
    </div>
    <div class="sidebar-stat-card">
        <div class="sidebar-stat-label">{t('unlocated_reports')}</div>
        <div class="sidebar-stat-val" style="color: #F59E0B;">{unlocated_count}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main Application Header & Portal Switcher
# ---------------------------------------------------------------------------
top_c1, top_c2 = st.columns([3, 1])
with top_c1:
    st.title(t("app_title"))
    st.caption(t("app_subtitle"))
with top_c2:
    portal = st.selectbox(
        t("portal_switcher"),
        options=[t("admin_portal"), t("citizen_portal")],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Portal Rendering
# ---------------------------------------------------------------------------
if portal == t("admin_portal"):
    # 8 Polish Administrative Tabs
    tabs = st.tabs([
        t("tab_overview"),
        t("tab_hotspot_map"),
        t("tab_cluster_explorer"),
        t("tab_priority_queue"),
        t("tab_emergency_panel"),
        t("tab_evidence"),
        t("tab_pattern_analysis"),
        t("tab_interventions"),
    ])

    # Tab 0: Executive Overview
    with tabs[0]:
        render_overview(
            clusters=filtered_clusters,
            complaints=filtered_complaints,
            interventions=pipeline.interventions,
            selected_region=selected_region,
        )

    # Tab 1: Incident Map & Complaint Density Heatmap
    with tabs[1]:
        render_hotspot_map(
            clusters=filtered_clusters,
            unlocated_complaints=pipeline.get_unlocated_complaints(selected_region),
            all_complaints=filtered_complaints,
        )

    # Tab 2: Semantic Cluster Explorer
    with tabs[2]:
        render_cluster_explorer(
            clusters=filtered_clusters,
            complaints=filtered_complaints,
            selected_region=selected_region,
        )

    # Tab 3: Deterministic Priority Queue + Optional Groq Context
    with tabs[3]:
        render_priority_queue(
            clusters=filtered_clusters,
            complaints=filtered_complaints,
        )

    # Tab 4: Emergency Hazard Detection Panel
    with tabs[4]:
        render_emergency_panel(
            clusters=filtered_clusters,
            complaints=filtered_complaints,
        )

    # Tab 5: Grounded Evidence Dossier
    with tabs[5]:
        render_evidence_panel(
            complaints=filtered_complaints,
            clusters=filtered_clusters,
        )

    # Tab 6: Recurrence & Systemic Pattern Analysis
    with tabs[6]:
        render_pattern_analysis(
            clusters=filtered_clusters,
            complaints=filtered_complaints,
        )

    # Tab 7: Intervention Tracking & Observational Evaluation
    with tabs[7]:
        render_intervention_view(
            interventions=pipeline.interventions,
            clusters=filtered_clusters,
            complaints=pipeline.enriched_complaints,
            pipeline=pipeline,
        )

else:
    # Citizen Portal — Clean complaint submission interface
    render_citizen_portal(pipeline)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("सुनवाई-AI (Sunwa-AI) · Multilingual Civic Grievance Intelligence Platform")
