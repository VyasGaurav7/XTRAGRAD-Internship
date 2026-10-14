"""
CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images
Full-stack interactive Streamlit application implementing Figures 2 through 7 from the research paper:
- Figure 2: Home Page (Dataset overview, paper context, navigation)
- Figure 3: User Registration Page (User metadata, input validation, local persistence)
- Figure 4: User Login Page (Authentication, demo accounts, session state)
- Figure 5: Main Dashboard (Unified multi-tab workspace: Prediction, Graph, Notebook, Signout)
- Figure 6: Upload Input Page (Model selector, image uploader, pre-loaded sample gallery)
- Figure 7: Prediction Result Page (Verdict, confidence, Grad-CAM visualization, forensic metrics)

Reference:
Bird, J.J. & Lotfi, A. (2024). CIFAKE: Image Classification and Explainable
Identification of AI-Generated Synthetic Images. IEEE Access / arXiv:2303.14126
"""

import os
import time
import json
import numpy as np
from PIL import Image
import streamlit as st

# Local modules
from models.model_loader import load_cifake_model, get_model_specs
from models.xai_engine import analyze_image_with_xai
from auth.user_manager import register_user, authenticate_user
from utils.chart_utils import (
    create_model_comparison_chart,
    create_confusion_matrix_figure,
    create_training_curves,
    create_roc_curve,
    create_dataset_distribution_chart
)


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAKE - AI Image Classification & Explainable AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ── Global Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Dark cyber-forensics theme */
.stApp {
    background-color: #0B0F19;
    color: #E2E8F0;
}

/* Header & Navigation Bar */
.cifake-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    background: #111827;
    border-radius: 12px;
    border: 1px solid #1F2937;
    margin-bottom: 2rem;
}
.brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    background: radial-gradient(circle at 50% 20%, rgba(99, 102, 241, 0.15) 0%, rgba(11, 15, 25, 0) 70%);
    border-radius: 20px;
    margin-bottom: 2rem;
}
.hero-heading {
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    background: linear-gradient(135deg, #F8FAFC 0%, #CBD5E1 50%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.hero-subheading {
    font-size: 1.15rem;
    color: #94A3B8;
    max-width: 800px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.6;
}

/* Metric / Stat Cards */
.stat-card {
    background: #131B2E;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    border-color: #38BDF8;
}
.stat-val {
    font-size: 1.85rem;
    font-weight: 800;
    color: #38BDF8;
}
.stat-label {
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* Verdict Badges */
.verdict-box {
    padding: 1.25rem;
    border-radius: 14px;
    text-align: center;
    margin: 1.25rem 0;
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: 0.02em;
}
.verdict-real {
    background: rgba(16, 185, 129, 0.12);
    border: 2px solid #10B981;
    color: #34D399;
}
.verdict-fake {
    background: rgba(239, 68, 68, 0.12);
    border: 2px solid #EF4444;
    color: #F87171;
}

/* Forensic Narrative Card */
.narrative-card {
    background: #111827;
    border-left: 4px solid #818CF8;
    border-radius: 8px;
    padding: 1.25rem;
    margin-top: 1rem;
    color: #CBD5E1;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Image grid containers */
.image-label {
    text-align: center;
    font-weight: 600;
    font-size: 0.85rem;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}

/* Streamlit button custom styles */
div.stButton > button:first-child {
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"  # Options: 'home', 'login', 'register', 'dashboard'

if "user" not in st.session_state:
    st.session_state.user = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "prediction"

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None


# ── Navigation Controller ────────────────────────────────────────────────────
def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()


# ── Global Top Navbar ────────────────────────────────────────────────────────
col_nav1, col_nav2, col_nav3 = st.columns([4, 2, 4])
with col_nav1:
    st.markdown('<div class="brand-title">🔬 CIFAKE FORENSICS PLATFORM</div>', unsafe_allow_html=True)

with col_nav3:
    nav_cols = st.columns(4)
    if st.session_state.user is None:
        if nav_cols[1].button("🏠 Home", use_container_width=True):
            navigate_to("home")
        if nav_cols[2].button("📝 Register", use_container_width=True):
            navigate_to("register")
        if nav_cols[3].button("🔑 Login", use_container_width=True):
            navigate_to("login")
    else:
        if nav_cols[1].button("📊 Dashboard", use_container_width=True):
            navigate_to("dashboard")
        nav_cols[2].markdown(f"<div style='padding-top:0.5rem; text-align:center; color:#38BDF8; font-weight:600;'>👤 {st.session_state.user['username']}</div>", unsafe_allow_html=True)
        if nav_cols[3].button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            navigate_to("home")

st.markdown("<hr style='border-color:#1F2937; margin-top:0.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-heading">CIFAKE: Image Classification & Explainable AI Identification</div>
        <div class="hero-subheading">
            Detecting AI-generated synthetic images from Latent Diffusion Models (Stable Diffusion v1.4)
            versus authentic photographs from CIFAR-10 with deep neural networks and Grad-CAM visual forensics.
        </div>
        <div style="display:flex; justify-content:center; gap:0.5rem; flex-wrap:wrap; margin-bottom:1.5rem;">
            <span style="background:rgba(99,102,241,0.2); color:#A5B4FC; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-weight:600; border:1px solid rgba(99,102,241,0.4);">📄 IEEE Access 2024</span>
            <span style="background:rgba(16,185,129,0.2); color:#6EE7B7; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-weight:600; border:1px solid rgba(16,185,129,0.4);">⭐ Modified CNN2D (95.78% Acc)</span>
            <span style="background:rgba(56,189,248,0.2); color:#7DD3FC; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-weight:600; border:1px solid rgba(56,189,248,0.4);">⚡ EfficientNetB0 (96.52% Acc)</span>
            <span style="background:rgba(244,114,182,0.2); color:#F9A8D4; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-weight:600; border:1px solid rgba(244,114,182,0.4);">🔍 Grad-CAM Explainable AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Statistics Grid
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    with stat_c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-val">120,000</div>
            <div class="stat-label">Total Images in CIFAKE</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-val">50% / 50%</div>
            <div class="stat-label">Balanced Real vs Synthetic</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-val">10 Classes</div>
            <div class="stat-label">Matching CIFAR-10 Semantics</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-val">95.78%</div>
            <div class="stat-label">Paper Proposed Modified CNN</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Feature & Architecture Highlights
    col_feat1, col_feat2 = st.columns(2, gap="large")
    with col_feat1:
        st.markdown("### 🎯 Problem Statement & Threat Vector")
        st.markdown("""
        The rapid advancement of **Latent Diffusion Models (LDMs)**, particularly Stable Diffusion, has made
        generating hyper-realistic synthetic media effortless. These synthetic images threaten:
        - **Digital Forensics & News Integrity**: Diffusion models can manufacture fabricated visual evidence.
        - **Biometric Security & Identity**: Deepfake imagery undermines identity verification.
        - **Intellectual Property**: Synthetic replicates obfuscate artistic provenance.

        **CIFAKE** tackles this challenge by systematically classifying images at 32×32 resolution and
        localizing the subtle latent frequency noise artifacts left during the reverse diffusion de-noising process.
        """)

        cta_col1, cta_col2 = st.columns(2)
        if cta_col1.button("🚀 Launch Live Detection", use_container_width=True):
            if st.session_state.user:
                navigate_to("dashboard")
            else:
                navigate_to("login")
        if cta_col2.button("📝 Register New Account", use_container_width=True):
            navigate_to("register")

    with col_feat2:
        st.markdown("### 🧠 Implemented Architectures & Innovation")
        st.markdown("""
        1. **Modified CNN2D (Paper Winner)**:
           - Replaces standard spatial `Flatten` with `GlobalAveragePooling2D` and `Dropout(0.3)`.
           - **Mitigates spatial noise overfitting**: prevents the neural network from memorizing position-specific diffusion noise.
           - Achieves **95.775% accuracy** on 20,000 test images (+0.885% over baseline).
        2. **Baseline CNN2D (Reference)**:
           - Two Conv2D blocks (128 filters, 3×3, ReLU) with MaxPool, Flatten, Dense(64) — **94.89% accuracy**.
        3. **EfficientNetB0 (Transfer Learning)**:
           - Pre-trained ImageNet compound-scaled network fine-tuned at 96×96 resolution — **96.52% accuracy**.
        4. **Explainable AI (Grad-CAM)**:
           - Computes class gradients on target convolutional feature maps to visually prove **why** an image is real or fake.
        """)

    st.write("")
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#64748B; font-size:0.85rem;">
        Reference: Bird, J.J. & Lotfi, A. (2024). <i>CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images</i>. IEEE Access.<br>
        Built with TensorFlow / Keras, Grad-CAM XAI Engine, and Streamlit.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: USER REGISTRATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "register":
    st.markdown("<h2 style='text-align:center;'>📝 User Registration</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94A3B8;'>Register to access the CIFAKE Deep Learning Forensics Dashboard</p>", unsafe_allow_html=True)

    reg_col1, reg_col2, reg_col3 = st.columns([1, 2, 1])
    with reg_col2:
        with st.form("registration_form"):
            reg_username = st.text_input("User Name*", placeholder="e.g. gaurav_vyas")
            reg_name = st.text_input("Full Name*", placeholder="e.g. Gaurav Vyas")
            reg_email = st.text_input("Email Address*", placeholder="e.g. gaurav@example.com")
            reg_mobile = st.text_input("Mobile Number*", placeholder="e.g. +91 9876543210")
            reg_password = st.text_input("Password*", type="password", placeholder="At least 6 characters")
            reg_confirm_pw = st.text_input("Confirm Password*", type="password")

            submitted = st.form_submit_button("Register Account", use_container_width=True)

            if submitted:
                if reg_password != reg_confirm_pw:
                    st.error("Passwords do not match. Please re-enter matching passwords.")
                else:
                    success, msg = register_user(
                        username=reg_username,
                        name=reg_name,
                        email=reg_email,
                        mobile=reg_mobile,
                        password=reg_password
                    )
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        navigate_to("login")
                    else:
                        st.error(msg)

        st.markdown("<div style='text-align:center; margin-top:1rem;'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("Go to Login Page", use_container_width=True):
            navigate_to("login")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: USER LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "login":
    st.markdown("<h2 style='text-align:center;'>🔑 User Login</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94A3B8;'>Authenticate to access real-time inference and XAI attribution</p>", unsafe_allow_html=True)

    login_col1, login_col2, login_col3 = st.columns([1, 2, 1])
    with login_col2:
        # Demo credentials helper
        st.info("💡 **Quick Demo Credentials**:\n- Username: `admin` | Password: `admin123`\n- Username: `researcher` | Password: `cifake2024`")

        with st.form("login_form"):
            login_username = st.text_input("User Name*", placeholder="Enter username")
            login_password = st.text_input("Password*", type="password", placeholder="Enter password")

            login_submitted = st.form_submit_button("Sign In to Dashboard", use_container_width=True)

            if login_submitted:
                success, result = authenticate_user(login_username, login_password)
                if success:
                    st.session_state.user = result
                    st.success(f"Welcome back, {result['name']}! Redirecting to Dashboard...")
                    time.sleep(0.8)
                    navigate_to("dashboard")
                else:
                    st.error(result)

        st.markdown("<div style='text-align:center; margin-top:1rem;'>Need a new account?</div>", unsafe_allow_html=True)
        if st.button("Register New User", use_container_width=True):
            navigate_to("register")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5, 6, 7: MAIN DASHBOARD & TABS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":
    # Ensure authenticated session
    if st.session_state.user is None:
        st.warning("Please sign in to access the Dashboard.")
        if st.button("Go to Login"):
            navigate_to("login")
        st.stop()

    st.markdown(f"""
    <div style="background:#131B2E; padding:1rem 1.5rem; border-radius:12px; border:1px solid #1E293B; margin-bottom:1.5rem; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-size:1.25rem; font-weight:700; color:#F8FAFC;">CIFAKE Forensics Workspace</span><br>
            <span style="color:#94A3B8; font-size:0.85rem;">Authenticated User: <b>{st.session_state.user['name']}</b> ({st.session_state.user['email']})</span>
        </div>
        <div>
            <span style="background:rgba(16,185,129,0.2); color:#6EE7B7; border:1px solid #10B981; padding:4px 10px; border-radius:6px; font-size:0.8rem; font-weight:600;">ACTIVE SESSION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Main Tabs as specified in Figure 5
    tab_pred, tab_graph, tab_note, tab_signout = st.tabs([
        "🔮 Prediction (Fig. 6 & 7)",
        "📈 Graph (Metrics & Benchmarks)",
        "📓 Notebook (Research Pipeline)",
        "🚪 Signout"
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: PREDICTION & XAI ATTRIBUTION (FIGURE 6 & 7)
    # ──────────────────────────────────────────────────────────────────────────
    with tab_pred:
        col_ctrl, col_display = st.columns([1, 2], gap="large")

        with col_ctrl:
            st.markdown("### ⚙️ Inference Configuration")

            # 1. Model Selector
            model_options = {
                "Modified_CNN2D": "⭐ Modified CNN2D (Trained on 100K CIFAKE Dataset - 89.2% Acc)",
                "EfficientNetB0": "⚡ EfficientNetB0 (Transfer Learning - 96.5% Benchmark)",
                "CNN2D_Baseline": "CNN2D Baseline (Standard Conv2D - Spatial Flatten)"
            }
            selected_model_key = st.selectbox(
                "Select Model Architecture:",
                options=list(model_options.keys()),
                format_func=lambda k: model_options[k],
                index=0
            )

            # Display Model Specifications
            specs = get_model_specs(selected_model_key)
            with st.expander("📋 Model Specifications & Benchmarks", expanded=True):
                st.markdown(f"**Architecture**: `{specs['name']}`")
                st.markdown(f"**Description**: {specs['description']}")
                m_c1, m_c2 = st.columns(2)
                m_c1.metric("Accuracy", f"{specs['accuracy']:.2f}%")
                m_c1.metric("Precision", f"{specs['precision']:.2f}%")
                m_c2.metric("Recall", f"{specs['recall']:.2f}%")
                m_c2.metric("F1-Score", f"{specs['f1']:.2f}%")
                st.caption(f"Input Shape: {specs['input_shape']} | Target Layer: `{specs['target_layer']}` | Params: {specs['params']}")

            # 2. XAI Controls
            st.markdown("### 🎨 Explainable AI Settings")
            colormap_name = st.selectbox("Grad-CAM Colormap", ["jet", "plasma", "viridis", "inferno", "magma", "coolwarm"])
            overlay_alpha = st.slider("Heatmap Overlay Transparency", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

            # 3. Input Source Selection
            st.markdown("### 🖼️ Input Image Selection")
            input_source = st.radio("Choose Input Mode:", ["📁 Upload Custom Image", "🖼️ Pre-loaded Sample Gallery"], horizontal=True)

            selected_image = None
            sample_caption = ""

            if input_source == "📁 Upload Custom Image":
                uploaded_file = st.file_uploader("Upload an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file).convert("RGB")
                    sample_caption = f"Uploaded File: {uploaded_file.name}"
            else:
                st.markdown("**Select from Verified Test Samples:**")
                sample_category = st.radio("Category Filter:", ["Real Photos (CIFAR-10)", "Synthetic (Stable Diffusion)"], horizontal=True)

                sample_base = "samples/real" if "Real" in sample_category else "samples/fake"
                if os.path.exists(sample_base):
                    sample_files = [f for f in os.listdir(sample_base) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    # Display 3x3 sample thumbnail picker
                    thumb_cols = st.columns(3)
                    for i, s_file in enumerate(sample_files[:6]):
                        s_path = os.path.join(sample_base, s_file)
                        with thumb_cols[i % 3]:
                            thumb_img = Image.open(s_path)
                            st.image(thumb_img, use_container_width=True)
                            btn_label = s_file.split(".")[0].replace("_", " ").title()
                            if st.button(f"Pick: {btn_label}", key=f"btn_sample_{s_file}", use_container_width=True):
                                st.session_state.selected_sample = s_path

                if st.session_state.selected_sample and os.path.exists(st.session_state.selected_sample):
                    selected_image = Image.open(st.session_state.selected_sample).convert("RGB")
                    sample_caption = f"Selected: {os.path.basename(st.session_state.selected_sample)}"

            analyze_btn = st.button("🔍 Run CIFAKE Classification & Grad-CAM", type="primary", use_container_width=True)

        # Right Column: Detection Results & Visual Attribution Grid
        with col_display:
            st.markdown("### 📊 Detection Verdict & Explainable AI Attribution")

            if selected_image is None:
                st.info("👈 Please upload an image or choose a pre-loaded sample from the left panel to begin forensic analysis.")
            else:
                if analyze_btn or "last_analysis" in st.session_state:
                    # Run inference and XAI
                    with st.spinner(f"Computing forward inference and Grad-CAM class activation on `{specs['name']}`..."):
                        start_time = time.time()
                        dl_model, base_dl, model_type = load_cifake_model(selected_model_key)
                        analysis = analyze_image_with_xai(
                            pil_image=selected_image,
                            model=dl_model,
                            model_type=model_type,
                            base_model=base_dl,
                            colormap_name=colormap_name,
                            alpha=overlay_alpha
                        )
                        inference_ms = (time.time() - start_time) * 1000.0

                    label = analysis["label"]
                    confidence = analysis["confidence"]
                    prob_real = analysis["prob_real"]
                    prob_fake = analysis["prob_fake"]
                    metrics = analysis["forensic_metrics"]

                    # 1. Verdict Banner (Figure 7)
                    if label == "REAL":
                        st.markdown("""
                        <div class="verdict-box verdict-real">
                            ✅ AUTHENTIC PHOTOGRAPH (CIFAR-10)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="verdict-box verdict-fake">
                            🚨 AI-GENERATED SYNTHETIC (STABLE DIFFUSION v1.4)
                        </div>
                        """, unsafe_allow_html=True)

                    # 2. Confidence & Probability Gauges
                    gauge_c1, gauge_c2, gauge_c3, gauge_c4 = st.columns(4)
                    gauge_c1.metric("Model Verdict", label)
                    gauge_c2.metric("Confidence Score", f"{confidence * 100.0:.2f}%")
                    gauge_c3.metric("Real Probability", f"{prob_real * 100.0:.2f}%")
                    gauge_c4.metric("Inference Latency", f"{inference_ms:.1f} ms")

                    st.write("")
                    # Dual Progress Bars
                    st.write("**Probability Distribution**:")
                    st.progress(prob_real, text=f"🟢 REAL Probability: {prob_real*100.0:.2f}%")
                    st.progress(prob_fake, text=f"🔴 FAKE Probability: {prob_fake*100.0:.2f}%")

                    st.write("")
                    st.markdown("#### 🔬 4-Way Visual Attribution Grid")

                    # 3. 4-Way Visual Attribution Grid
                    g_c1, g_c2, g_c3, g_c4 = st.columns(4)
                    with g_c1:
                        st.markdown("<div class='image-label'>1. Original Input</div>", unsafe_allow_html=True)
                        st.image(selected_image, use_container_width=True)
                    with g_c2:
                        st.markdown("<div class='image-label'>2. Grad-CAM Heatmap</div>", unsafe_allow_html=True)
                        st.image(analysis["heatmap_colored"], use_container_width=True)
                    with g_c3:
                        st.markdown("<div class='image-label'>3. Superimposed Overlay</div>", unsafe_allow_html=True)
                        st.image(analysis["overlay_image"], use_container_width=True)
                    with g_c4:
                        st.markdown("<div class='image-label'>4. Grayscale Saliency</div>", unsafe_allow_html=True)
                        st.image(analysis["saliency_grayscale"], use_container_width=True)

                    # 4. Forensic Attribution Report (Figure 7)
                    st.write("")
                    st.markdown("#### 📑 Forensic Attribution & Diagnostic Observations")

                    f_c1, f_c2, f_c3 = st.columns(3)
                    f_c1.metric("Central Subject Attention", f"{metrics['center_attention_pct']:.1f}%")
                    f_c2.metric("Peripheral/Background Attention", f"{metrics['background_attention_pct']:.1f}%")
                    f_c3.metric("High-Activation Hotspot Area", f"{metrics['hotspot_area_pct']:.1f}%")

                    st.markdown(f"""
                    <div class="narrative-card">
                        <b>Forensic Findings Narrative</b>:<br>
                        {metrics['narrative']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Export Forensic Report
                    report_payload = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": specs["name"],
                        "target_layer": specs["target_layer"],
                        "verdict": label,
                        "confidence_pct": round(confidence * 100.0, 2),
                        "prob_real": round(prob_real, 4),
                        "prob_fake": round(prob_fake, 4),
                        "latency_ms": round(inference_ms, 2),
                        "metrics": metrics
                    }
                    st.download_button(
                        label="📥 Download Forensic Audit Report (JSON)",
                        data=json.dumps(report_payload, indent=2),
                        file_name=f"cifake_forensic_report_{int(time.time())}.json",
                        mime="application/json"
                    )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: GRAPH (EVALUATION & BENCHMARKS)
    # ──────────────────────────────────────────────────────────────────────────
    with tab_graph:
        st.markdown("### 📈 Comprehensive Model Benchmarking & Statistical Evaluation")
        st.markdown("Visualizing empirical findings from the CIFAKE research paper (Table 1: 20,000 Test Images).")

        # Top Bar Chart: Comparative Performance
        fig_comp = create_model_comparison_chart()
        st.plotly_chart(fig_comp, use_container_width=True)

        g_col1, g_col2 = st.columns(2, gap="medium")
        with g_col1:
            st.markdown("#### 🎯 Confusion Matrix on 20,000 Test Set")
            cm_model_choice = st.selectbox("Select Model for Confusion Matrix:", ["Modified_CNN2D", "EfficientNetB0", "CNN2D_Baseline"], key="cm_select")
            fig_cm = create_confusion_matrix_figure(cm_model_choice)
            st.plotly_chart(fig_cm, use_container_width=True)

        with g_col2:
            st.markdown("#### 📉 Receiver Operating Characteristic (ROC)")
            fig_roc = create_roc_curve()
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("#### 🔄 Training Convergence & Loss Minimization")
        fig_train = create_training_curves()
        st.plotly_chart(fig_train, use_container_width=True)

        st.markdown("#### 📊 Dataset Semantic Distribution Across Classes")
        fig_dist = create_dataset_distribution_chart()
        st.plotly_chart(fig_dist, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: NOTEBOOK (RESEARCH METHODOLOGY & PIPELINE)
    # ──────────────────────────────────────────────────────────────────────────
    with tab_note:
        st.markdown("### 📓 Research Notebook & Methodology Walkthrough")
        st.markdown("""
        This interactive walkthrough summarizes the exact methodology and empirical discoveries
        formulated in the attached research paper:
        *Bird, J.J. & Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.*
        """)

        with st.expander("1️⃣ Problem Definition & The CIFAKE Dataset", expanded=True):
            st.markdown("""
            - **Problem**: Modern latent diffusion architectures (Stable Diffusion v1.4) de-noise Gaussian latent representations to generate photorealistic imagery across complex categories.
            - **Dataset Construction**:
              - Real Images: 60,000 authentic images sampled from the classical **CIFAR-10** benchmark.
              - Synthetic Images: 60,000 images prompted through **Stable Diffusion v1.4** using class-conditional prompt prefixes.
              - 10 semantic classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
              - Scale: 100,000 training images (50,000 real / 50,000 fake) and 20,000 test images (10,000 real / 10,000 fake).
            """)

        with st.expander("2️⃣ Architectural Breakthrough: Mitigating Spatial Overfitting with GAP"):
            st.markdown("""
            A critical empirical finding of the paper is the difference between **Baseline CNN2D** and **Modified CNN2D**:
            - **Baseline CNN2D** utilizes a spatial `Flatten` layer. The fully connected dense layer memorizes spatial coordinates of diffusion artifacts, leading to subtle spatial overfitting on local noise coordinates.
            - **Modified CNN2D** introduces `GlobalAveragePooling2D` (GAP) followed by `Dropout(0.3)`.
              - GAP enforces structural feature invariance across the 2D feature map.
              - Reduces total parameter count from **446,209** down to **180,481** (a 59.5% parameter reduction!).
              - Improves overall test accuracy from **94.89%** to **95.775%**.
            """)

        with st.expander("3️⃣ Mathematical Formulation of Explainable AI (Grad-CAM)"):
            st.markdown(r"""
            Gradient-weighted Class Activation Mapping (Grad-CAM) calculates the gradient of the class prediction score $y^c$
            with respect to the feature map activations $A^k$ of the final convolutional layer:

            $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$

            The final coarse localization heatmap $L_{\text{Grad-CAM}}^c$ is computed by taking a weighted combination of forward feature maps
            followed by a Rectified Linear Unit (ReLU) operation to isolate features that have a positive influence on the target class:

            $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

            This allows us to visually inspect whether the model focuses on the semantic object (eyes, wings, wheels) or on unnatural background diffusion noise!
            """)

        with st.expander("4️⃣ Summary Benchmark Results (Paper Table 1)"):
            st.table([
                {"Model": "Baseline CNN2D", "Accuracy": "94.890%", "Precision": "94.921%", "Recall": "94.896%", "F1-Score": "94.889%", "Parameters": "446,209"},
                {"Model": "Modified CNN2D (Winner ⭐)", "Accuracy": "95.775%", "Precision": "95.830%", "Recall": "95.768%", "F1-Score": "95.773%", "Parameters": "180,481"},
                {"Model": "EfficientNetB0 (SOTA)", "Accuracy": "96.520%", "Precision": "96.580%", "Recall": "96.490%", "F1-Score": "96.535%", "Parameters": "4,213,668"},
            ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4: SIGNOUT
    # ──────────────────────────────────────────────────────────────────────────
    with tab_signout:
        st.markdown("### 🚪 Sign Out")
        st.markdown(f"Are you sure you want to end the session for **{st.session_state.user['name']}**?")
        if st.button("Confirm Logout", type="primary"):
            st.session_state.user = None
            st.session_state.page = "home"
            st.success("You have been signed out successfully.")
            time.sleep(0.5)
            st.rerun()
