"""Sunwa-AI — Citizen Grievance Portal Component.
Separate from administrative interface. No admin data exposed.
No image upload. Generates complaint reference ID.
"""
import streamlit as st
from locales.i18n import t

REGIONS = ["Nerul", "Panvel", "Vashi", "Kharghar", "Belapur"]


def render_citizen_portal(pipeline) -> None:
    # Citizen portal header
    st.markdown(
        """
        <div style="text-align: center; padding: 32px 0 16px;">
            <h1 style="font-size: 2.2rem; font-weight: 800;
                background: linear-gradient(90deg, #38BDF8, #818CF8);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                सुनवाई-AI
            </h1>
            <h3 style="color: #94A3B8; font-weight: 400; margin-top: -8px;">
                Citizen Grievance Portal
            </h3>
            <p style="color: #64748B; max-width: 560px; margin: 8px auto;">
                Submit your civic complaint in English, हिंदी, or मराठी.
                Your complaint will be reviewed by the municipal administration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Form
    with st.form("citizen_complaint_form", clear_on_submit=False):
        st.markdown("### 📋 Submit Your Complaint")

        region = st.selectbox(
            t("region_select"),
            ["— Select Region —"] + REGIONS,
        )
        complaint_text = st.text_area(
            t("complaint_text_label"),
            height=140,
            placeholder=(
                "Describe the issue in detail.\n"
                "आप हिंदी में लिख सकते हैं।\n"
                "मराठीत लिहिणे शक्य आहे।"
            ),
        )
        location_hint = st.text_input(
            t("location_hint_label"),
            placeholder="e.g. Near SIES College gate / नेरुल स्टेशन जवळ",
        )

        submitted = st.form_submit_button(t("submit_button"), type="primary", use_container_width=True)

    if submitted:
        errors = []
        if region == "— Select Region —":
            errors.append(t("validation_region"))
        if not complaint_text.strip():
            errors.append(t("validation_empty"))
        if len(complaint_text.strip()) < 10:
            errors.append("Complaint must be at least 10 characters.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Submitting your complaint..."):
                try:
                    raw = pipeline.submit_complaint(
                        region=region,
                        text=complaint_text.strip(),
                        location_hint=location_hint.strip(),
                    )
                    st.success(f"✅ {t('submission_success')}")
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #38BDF8; border-radius: 12px;
                            padding: 20px; text-align: center; margin-top: 16px;
                            background: rgba(56,189,248,0.05);">
                            <h4 style="color:#38BDF8; margin-bottom: 8px;">
                                {t('reference_id')}
                            </h4>
                            <h2 style="color:#F8FAFC; font-family: monospace; letter-spacing: 2px;">
                                {raw.complaint_id}
                            </h2>
                            <p style="color:#94A3B8; font-size: 0.9rem; margin-top: 8px;">
                                {t('submission_note')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    st.markdown("**Your submitted complaint:**")
                    st.markdown(
                        f'<blockquote style="border-left: 3px solid #38BDF8; padding: 8px 16px; '
                        f'color: #CBD5E1;">{complaint_text.strip()}</blockquote>',
                        unsafe_allow_html=True,
                    )
                    if location_hint.strip():
                        st.caption(f"📍 Location hint: {location_hint.strip()}")
                    st.caption(f"🕐 Submitted at: {raw.submitted_at[:19]} UTC")
                except Exception as exc:
                    st.error(f"Submission failed: {exc}. Please try again.")

    st.markdown("---")
    st.caption(
        "सुनवाई-AI is a civic technology demonstration system. "
        "Complaints are processed by automated analysis and reviewed by administrators. "
        "No personal data is stored beyond the complaint text and reference ID."
    )
