"""Sunwa-AI — Grounded Evidence Dossier Component.
Displays original complaint excerpts, deterministic scoring breakdown,
safety signals, pattern classification rationale, and Groq contextual interpretation.
Never fabricates evidence or confidence scores.
Handles missing optional fields safely.
"""
import streamlit as st
from typing import List
from locales.i18n import t
from models.schemas import IncidentCluster, EnrichedComplaint, LocationStatus


def _loc_status_badge(status: str) -> str:
    badges = {
        "resolved": "✅ Resolved (Verified Landmark)",
        "ambiguous": "⚠️ Ambiguous (Multiple Candidates)",
        "region_only": "🗺️ Region-Level Only",
        "unresolved": "❌ Unresolved (Pending Review)",
    }
    return badges.get(status, f"❓ {status}")


def render_evidence_panel(
    complaints: List[EnrichedComplaint],
    clusters: List[IncidentCluster],
) -> None:
    st.markdown(f"### 📁 {t('tab_evidence')}")
    st.caption(
        "Complete grounded evidence audit trail for each incident cluster. "
        "Examines original citizen submissions, deterministic formula calculations, "
        "geographic resolution methods, and qualitative analysis."
    )

    if not clusters:
        st.info(t("no_data"))
        return

    # Cluster selector
    cluster_options = {
        f"{c.cluster_id} — {c.landmark_name} [{c.priority.priority_level}] ({c.complaint_count} reports)": c
        for c in clusters
    }
    selected_key = st.selectbox("Select Incident Cluster for Evidence Audit", list(cluster_options.keys()))
    if not selected_key:
        return

    cluster = cluster_options[selected_key]
    p = cluster.priority
    rec = cluster.recommendation
    cluster_complaints = [c for c in complaints if c.complaint_id in cluster.complaint_ids]

    # ── Cluster Header ─────────────────────────────────────────────────
    priority_color = {
        "Critical": "#EF4444",
        "High": "#F97316",
        "Medium": "#EAB308",
        "Low": "#22C55E",
    }.get(p.priority_level, "#60A5FA")

    st.markdown(
        f"""<div style="border-left: 4px solid {priority_color}; padding: 14px 18px; border-radius: 8px;
            background: rgba(255,255,255,0.03); margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h3 style="margin:0; color: {priority_color};">{cluster.landmark_name}</h3>
                <span style="font-size: 1.1rem; font-weight: 700; color: #F1F5F9;">
                    Score: {p.overall_score}/100 ({p.priority_level})
                </span>
            </div>
            <p style="margin:6px 0 0; color:#94A3B8;">
                Cluster ID: <code>{cluster.cluster_id}</code> · Region: <b>{cluster.region}</b> ·
                Category: <b>{cluster.category}</b> · Pattern: <b>{cluster.pattern_type}</b>
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Deterministic Priority", f"{p.overall_score}/100")
    m2.metric("Severity Level", p.priority_level)
    m3.metric("Complaint Count", cluster.complaint_count)
    m4.metric("Incident Window", cluster.time_range)

    st.markdown("---")

    # ── Section 1: Deterministic Score Breakdown ───────────────────────
    st.markdown("#### 1. 📊 Deterministic Score & Factor Breakdown")
    st.caption("P = 0.25·S + 0.20·A + 0.15·D + 0.25·R + 0.15·T (Configurable weights, no LLM score override)")
    st.code(p.formula_expression, language=None)

    score_cols = st.columns(5)
    score_cols[0].metric("Severity (S)", f"{p.severity_score:.0f}/100")
    score_cols[1].metric("Population (A)", f"{p.affected_pop_score:.0f}/100")
    score_cols[2].metric("Duration (D)", f"{p.duration_score:.0f}/100")
    score_cols[3].metric("Safety (R)", f"{p.safety_risk_score:.0f}/100")
    score_cols[4].metric("Recurrence (T)", f"{p.recurrence_score:.0f}/100")

    for line in p.explanation_breakdown:
        st.caption(f"• {line}")

    if p.safety_override_triggered:
        st.error(f"🛡️ **Safety Override Applied:** {p.override_reason}")

    st.markdown("---")

    # ── Section 2: Pattern Recognition & Justification ────────────────
    st.markdown("#### 2. 🔄 Pattern Recognition & Analytical Hypothesis")
    st.info(
        f"**Classified Pattern:** `{cluster.pattern_type}`  \n"
        f"**Grounded Rationale:** Based on {cluster.complaint_count} reports across span {cluster.time_range}.  \n"
        f"**Hypothesis Note:** Observational classification. Requires on-ground inspection to verify root cause."
    )

    st.markdown("---")

    # ── Section 3: Geographic Grounding ────────────────────────────────
    st.markdown("#### 3. 📍 Geographic Resolution & Landmark Catalogue Match")
    if cluster_complaints:
        rep = cluster_complaints[0]
        loc = rep.resolved_location
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(f"**Resolution Status:** {_loc_status_badge(loc.location_status)}")
            st.markdown(f"**Matched Landmark:** `{loc.matched_landmark or 'Unresolved'}`")
            st.markdown(f"**Match Type / Engine:** `{loc.match_type}` (Confidence: {loc.match_score:.1f}%)")
            st.markdown(f"**Resolution Method:** {loc.resolution_reason}")
        with col_g2:
            if loc.latitude is not None and loc.longitude is not None:
                st.markdown(f"**Centroid Coordinates:** `{loc.latitude:.4f}, {loc.longitude:.4f}`")
                st.caption("📌 *Landmark-level precision from verified catalogue. Not exact GPS asset pin.*")
            else:
                st.warning("⚠️ *No valid coordinates resolved. Preserved for manual location review.*")
    else:
        st.caption("No complaint records linked.")

    st.markdown("---")

    # ── Section 4: AI Contextual Brief (Groq) ─────────────────────────
    if cluster.ai_interpretation:
        st.markdown("#### 4. 🤖 AI Contextual Interpretation (Groq)")
        st.caption("Qualitative contextual analysis. Clearly separated from deterministic numerical priority.")
        st.markdown(
            f"""<div style="border-left: 3px solid #818CF8; padding: 12px 16px; border-radius: 6px;
                background: rgba(129, 140, 248, 0.05); margin-bottom: 12px;">
                <span style="color: #A5B4FC; font-weight: 700; font-size: 0.85rem;">
                    🤖 AI-GENERATED (GROQ — GROUNDED IN EVIDENCE)
                </span>
                <div style="margin-top: 8px; color: #F1F5F9; white-space: pre-wrap; font-size: 0.95rem;">
{cluster.ai_interpretation}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("---")

    # ── Section 5: Grounded Complaint Excerpts ─────────────────────────
    st.markdown(f"#### 5. 📝 Original Complaint Excerpts ({len(cluster_complaints)})")
    st.caption("Original submitted text preserved verbatim across languages (English, Hindi, Marathi). No synthesized data.")

    for c in cluster_complaints:
        loc = c.resolved_location
        flags_badge = f" | 🚩 Flags: {', '.join(c.safety_flags)}" if c.safety_flags else ""
        with st.container():
            st.markdown(
                f"""<div style="border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
                    padding: 10px 14px; margin-bottom: 8px; background: rgba(255,255,255,0.02);">
                    <small style="color:#64748B;">
                        <b>{c.complaint_id}</b> · {c.region} · {c.submitted_at[:16]}
                        · Lang: <b>{c.detected_language}</b>
                        · Loc: {_loc_status_badge(loc.location_status)}{flags_badge}
                    </small><br>
                    <span style="color:#E2E8F0; font-size: 0.95rem;">\"{c.complaint_text}\"</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Section 6: Missing Information & Uncertainty ───────────────────
    st.markdown("---")
    st.markdown("#### 6. ⚠️ Missing Information & Uncertainty Analysis")
    missing_items = []
    if cluster.latitude is None:
        missing_items.append("Exact landmark coordinates unconfirmed (manual geotagging needed)")
    if cluster.complaint_count < 3:
        missing_items.append("Low complaint volume (statistical sample size is small)")
    if not any(c.safety_flags for c in cluster_complaints):
        missing_items.append("No safety keywords detected (standard maintenance protocol applies)")
    if rec.missing_info:
        missing_items.extend(rec.missing_info)

    if missing_items:
        for item in missing_items:
            st.caption(f"• ⚠️ {item}")
    else:
        st.caption("✅ All primary operational attributes resolved with adequate certainty.")
