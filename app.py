"""
SMARTPOSE – AI POWERED POSE SUGGESTION AND ALIGNMENT SYSTEM
Full-featured Computer Vision Application
Built with Streamlit, OpenCV, MediaPipe Tasks, and NumPy.
"""

import os
import time
import datetime
import cv2
import numpy as np
import streamlit as st

# Modular Architecture Imports
from pose_detector import PoseDetector, LANDMARKS
from pose_comparator import PoseComparator, ComparisonResult
from reference_manager import ReferencePoseManager, PREDEFINED_POSES
from auth import render_auth_page


# -----------------------------------------------------------------------------
# Page Configuration & Global Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartPose – AI Pose Assistant",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5 0%, #00C9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #546E7A;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1.4rem;
    }

    .feature-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -1px rgba(0, 0, 0, 0.08);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #64748B;
        line-height: 1.4;
    }

    .score-banner {
        border-radius: 12px;
        padding: 16px 20px;
        color: #FFFFFF;
        font-weight: 700;
        text-align: center;
        margin-bottom: 12px;
    }
    .score-excellent {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    }
    .score-good {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
    }
    .score-moderate {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    }
    .score-poor {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    }

    .badge-aligned {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-correction {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }

    .suggestion-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #92400E;
        font-weight: 500;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Home Dashboard"

if "sidebar_nav" not in st.session_state:
    st.session_state.sidebar_nav = "🏠 Home Dashboard"


def change_page(page_name: str):
    """Callback executed on button clicks before widgets are instantiated."""
    st.session_state["sidebar_nav"] = page_name
    st.session_state.nav_page = page_name


def select_pose_and_navigate(pose_name: str):
    """Callback to select reference pose and switch to live analysis."""
    st.session_state.selected_pose_name = pose_name
    st.session_state["sidebar_nav"] = "🎯 Live Pose Analysis"
    st.session_state.nav_page = "🎯 Live Pose Analysis"

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "selected_pose_name" not in st.session_state:
    st.session_state.selected_pose_name = "T-Pose"

if "session_captures" not in st.session_state:
    st.session_state.session_captures = []

if "latest_comparison" not in st.session_state:
    st.session_state.latest_comparison = None

if "custom_pose_data" not in st.session_state:
    st.session_state.custom_pose_data = None

# If user is not authenticated, render the dedicated Authentication Page
if not st.session_state.authenticated:
    render_auth_page()
    st.stop()



# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_pose_detector():
    """Initializes and caches the PoseDetector singleton."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "pose_landmarker_full.task")
    return PoseDetector(model_path=model_path)


@st.cache_resource
def get_reference_manager():
    """Initializes and caches the ReferencePoseManager."""
    return ReferencePoseManager()


@st.cache_resource
def get_pose_comparator():
    """Initializes and caches the PoseComparator."""
    return PoseComparator(tolerance_deg=15.0, warning_deg=30.0)


# -----------------------------------------------------------------------------
# Navigation & Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/yoga.png",
        width=64,
    )
    st.markdown("### **SMARTPOSE**")
    st.caption("AI Pose Suggestion & Alignment System")

    # User Profile & Sign Out
    if st.session_state.get("authenticated", False) and st.session_state.get("user"):
        u = st.session_state.user
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px; margin-top: 8px; margin-bottom: 8px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 600;">LOGGED IN AS</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #1E293B;">👤 {u.get('name', 'Athlete')}</div>
                <div style="font-size: 0.8rem; color: #0284C7;">@{u.get('username', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.camera_running = False
            st.rerun()

    st.markdown("---")

    nav_options = [
        "🏠 Home Dashboard",
        "🎯 Live Pose Analysis",
        "🖼️ Reference Pose Gallery",
        "📊 Results & History",
    ]

    selected_page = st.radio(
        "Navigate",
        nav_options,
        key="sidebar_nav",
    )
    st.session_state.nav_page = selected_page

    st.markdown("---")
    st.markdown("### ⚙️ Camera Settings")
    camera_index = st.number_input(
        "Webcam Index",
        min_value=0,
        max_value=4,
        value=0,
        step=1,
        help="0 is the primary built-in camera. Increase if using an external USB camera.",
    )
    mirror_camera = st.checkbox("Mirror Camera (Selfie Mode)", value=True)
    show_skeleton_overlay = st.checkbox("Show Skeleton Connections", value=True)
    show_angle_tags = st.checkbox("Display Joint Angles", value=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #94A3B8; text-align: center;'>"
        "SmartPose CV System v1.0<br>Powered by MediaPipe & Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if st.session_state.nav_page == "🏠 Home Dashboard":
    st.markdown('<div class="main-title">SMARTPOSE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">AI-Powered Human Pose Detection, Comparative Alignment, and Real-Time Correction System.</div>',
        unsafe_allow_html=True,
    )

    # Hero Banner
    hero_col1, hero_col2 = st.columns([3, 2])
    with hero_col1:
        st.markdown(
            """
            Welcome to **SmartPose**, an intelligent computer vision assistant that tracks your full-body posture 
            in real time using Google MediaPipe's 33 anatomical landmarks. 

            Compare your live movement against **5 professional yoga and fitness benchmark poses** (or upload your 
            own custom reference image), receive immediate biomechanical angle similarity feedback, and get 
            voice-like corrective guidance to perfect your form safely.
            """
        )

        c1, c2 = st.columns([1.5, 2.5])
        with c1:
            st.button(
                "🚀 Start Pose Analysis",
                type="primary",
                use_container_width=True,
                on_click=change_page,
                args=("🎯 Live Pose Analysis",),
            )
        with c2:
            st.button(
                "🖼️ Browse Reference Poses",
                use_container_width=True,
                on_click=change_page,
                args=("🖼️ Reference Pose Gallery",),
            )

    with hero_col2:
        st.info(
            "⚡ **System Status: Online**\n\n"
            "- **Inference Model:** MediaPipe Tasks PoseLandmarker Full (33 Landmarks)\n"
            "- **Camera Engine:** OpenCV VideoStream (Direct Hardware Pipe)\n"
            "- **Alignment Engine:** Trigonometric Multi-Joint Angular Deviation\n"
            "- **Reference Library:** 5 Certified Benchmark Poses + Dynamic Uploader"
        )

    st.markdown("---")
    st.subheader("💡 Core System Capabilities")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">👁️</div>
                <div class="feature-title">Live 33-Landmark Detection</div>
                <div class="feature-desc">
                    Captures 3D spatial coordinates across head, shoulders, elbows, wrists, hips, knees, and ankles with low CPU latency.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📐</div>
                <div class="feature-title">Precise Joint Angle Comparison</div>
                <div class="feature-desc">
                    Calculates exact multi-segment angles for upper body, lower limbs, and spine verticality, independent of body scale.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Real-Time Alignment Scoring</div>
                <div class="feature-desc">
                    Generates a calibrated 0–100% similarity score with color-coded skeleton highlights (Green, Amber, Red).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🗣️</div>
                <div class="feature-title">Smart Actionable Suggestions</div>
                <div class="feature-desc">
                    Instant, simple instructions: 'Raise your left arm', 'Straighten your knees', 'Square your shoulders'.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🖼️</div>
                <div class="feature-title">Custom Pose Ingestion</div>
                <div class="feature-desc">
                    Upload any photo of a trainer, athlete, or personal goal pose. SmartPose automatically extracts its target angles.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📸</div>
                <div class="feature-title">Snapshot & Session History</div>
                <div class="feature-desc">
                    Capture annotated snapshots of your best alignments, review diagnostic breakdowns, and download results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2: LIVE POSE ANALYSIS
# =============================================================================
elif st.session_state.nav_page == "🎯 Live Pose Analysis":
    st.markdown('<div class="main-title">🎯 Live Pose Analysis & Alignment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Mirror the reference target on the left; SmartPose analyzes your form in real-time on the right.</div>',
        unsafe_allow_html=True,
    )

    ref_mgr = get_reference_manager()
    detector = get_pose_detector()
    comparator = get_pose_comparator()

    # Pose Selection Bar
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 1.2, 1.2])
    with ctrl_col1:
        pose_options = list(PREDEFINED_POSES.keys()) + ["➕ Upload Custom Pose Image"]
        chosen_pose = st.selectbox(
            "Select Target Reference Pose:",
            pose_options,
            index=pose_options.index(st.session_state.selected_pose_name)
            if st.session_state.selected_pose_name in pose_options
            else 0,
        )
        st.session_state.selected_pose_name = chosen_pose

    with ctrl_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        start_camera = st.button("▶ Start Camera", type="primary", use_container_width=True)
    with ctrl_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        stop_camera = st.button("⏹ Stop Camera", use_container_width=True)

    if start_camera:
        st.session_state.camera_running = True
    if stop_camera:
        st.session_state.camera_running = False

    # Handle Custom Image Upload if selected
    target_angles = None
    ref_display_img = None
    pose_description = ""
    pose_category = ""

    if chosen_pose == "➕ Upload Custom Pose Image":
        upload_file = st.file_uploader(
            "Upload reference pose (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload an image showing a clear full-body posture.",
        )
        if upload_file is not None:
            bytes_data = upload_file.read()
            ok, annotated_custom, angles, msg = ref_mgr.process_custom_upload(detector, bytes_data)
            if ok:
                st.success(msg)
                ref_display_img = annotated_custom
                target_angles = angles
                pose_description = f"Custom Pose: {upload_file.name}"
                pose_category = "User Upload"
                st.session_state.custom_pose_data = {
                    "image": annotated_custom,
                    "angles": angles,
                    "name": upload_file.name,
                }
            else:
                st.error(msg)
                if annotated_custom is not None:
                    ref_display_img = annotated_custom
        elif st.session_state.custom_pose_data:
            ref_display_img = st.session_state.custom_pose_data["image"]
            target_angles = st.session_state.custom_pose_data["angles"]
            pose_description = f"Custom Pose: {st.session_state.custom_pose_data['name']}"
            pose_category = "User Upload"
    else:
        pose_info = ref_mgr.get_pose_by_name(chosen_pose)
        if pose_info:
            target_angles = pose_info["target_angles"]
            ref_display_img = ref_mgr.get_reference_image(chosen_pose)
            pose_description = pose_info["description"]
            pose_category = pose_info["category"]

    # Side-by-side Viewports (Reference vs Live Webcam)
    view_col1, view_col2 = st.columns([1, 1])

    with view_col1:
        st.markdown(f"#### 🖼️ Reference Target: **{chosen_pose}**")
        st.caption(f"**Category:** {pose_category} | {pose_description}")
        ref_placeholder = st.empty()
        if ref_display_img is not None:
            rgb_ref = cv2.cvtColor(ref_display_img, cv2.COLOR_BGR2RGB)
            ref_placeholder.image(rgb_ref, use_container_width=True)
        else:
            ref_placeholder.info("Select or upload a reference pose to view target.")

    with view_col2:
        st.markdown("#### 📷 Live Camera Feed & Skeleton Tracking")
        camera_placeholder = st.empty()
        live_alert_placeholder = st.empty()

    st.markdown("---")

    # Real-Time Alignment Analytics Area
    st.subheader("📊 Live Alignment Diagnostics & Corrections")

    score_col, sugg_col = st.columns([1, 1.5])
    with score_col:
        score_placeholder = st.empty()
        progress_placeholder = st.empty()
    with sugg_col:
        suggestions_placeholder = st.empty()

    aligned_col, fix_col = st.columns(2)
    with aligned_col:
        aligned_placeholder = st.empty()
    with fix_col:
        fix_placeholder = st.empty()

    # Capture Action Bar
    cap_c1, cap_c2 = st.columns([1.5, 3])
    with cap_c1:
        capture_btn = st.button("📸 Capture & Save Result Snapshot", use_container_width=True)
    with cap_c2:
        capture_status_placeholder = st.empty()

    # ---------------------------------------------------------
    # Camera Stream Execution Loop
    # ---------------------------------------------------------
    if not st.session_state.camera_running:
        camera_placeholder.info(
            "Camera is currently **idle**. Click **'▶ Start Camera'** above to begin live pose analysis."
        )
        score_placeholder.markdown(
            '<div class="score-banner score-good">Standby — Waiting for Camera Launch</div>',
            unsafe_allow_html=True,
        )
        suggestions_placeholder.info("Real-time form suggestions will appear here once the camera is running.")
        aligned_placeholder.markdown("**Aligned Joints (±15°):** —")
        fix_placeholder.markdown("**Joints Needing Correction:** —")
    else:
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.session_state.camera_running = False
            camera_placeholder.error(
                f"❌ Failed to open webcam at index `{camera_index}`.\n\n"
                "**Checklist:**\n"
                "- Verify camera is plugged in.\n"
                "- Ensure no other app (Zoom, Teams, Discord) is locking the webcam.\n"
                "- Try changing the 'Webcam Index' in the sidebar to 1 or 2."
            )
        else:
            prev_time = time.time()
            fps = 0.0

            try:
                while st.session_state.camera_running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        live_alert_placeholder.warning("⚠️ Warning: Empty frame received from webcam.")
                        break

                    if mirror_camera:
                        frame = cv2.flip(frame, 1)

                    h, w = frame.shape[:2]
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detection_result = detector.process_frame(rgb_frame)

                    has_pose = bool(
                        detection_result
                        and detection_result.pose_landmarks
                        and len(detection_result.pose_landmarks) > 0
                    )

                    comparison = None
                    annotated_frame = frame.copy()

                    if has_pose:
                        landmarks = detection_result.pose_landmarks[0]
                        lm_dict = detector.extract_landmark_dict(landmarks, w, h)
                        live_angles = detector.extract_joint_angles(lm_dict)

                        # Compare with target reference
                        if target_angles:
                            comparison = comparator.compare_poses(live_angles, target_angles)
                            st.session_state.latest_comparison = comparison

                            # Extract joint status dictionary for color-coded rendering
                            joint_status = {
                                k: comp.status for k, comp in comparison.joint_results.items()
                            }
                        else:
                            joint_status = {}

                        if show_skeleton_overlay:
                            detector.draw_annotated_skeleton(
                                annotated_frame,
                                landmarks,
                                joint_status=joint_status,
                                draw_angles=show_angle_tags,
                                angles=live_angles,
                            )

                    # Compute FPS
                    curr_time = time.time()
                    fps = 0.85 * fps + 0.15 * (1.0 / (curr_time - prev_time + 1e-6))
                    prev_time = curr_time

                    # Top FPS Tag on camera frame
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Render Camera frame in Streamlit
                    frame_disp = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_disp, channels="RGB", use_container_width=True)

                    # Update Analytics Placeholders
                    if has_pose and comparison:
                        live_alert_placeholder.empty()
                        score = comparison.overall_score

                        # Color class for score banner
                        if score >= 90:
                            s_class = "score-excellent"
                        elif score >= 75:
                            s_class = "score-good"
                        elif score >= 55:
                            s_class = "score-moderate"
                        else:
                            s_class = "score-poor"

                        score_placeholder.markdown(
                            f'<div class="score-banner {s_class}">'
                            f'<div style="font-size: 2.2rem; font-weight: 800;">{score}%</div>'
                            f'<div>{comparison.grade}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        progress_placeholder.progress(min(1.0, max(0.0, score / 100.0)))

                        # Form suggestions box
                        sugg_html = "<div>"
                        for s in comparison.suggestions:
                            sugg_html += f'<div class="suggestion-box">👉 {s}</div>'
                        sugg_html += "</div>"
                        suggestions_placeholder.markdown(sugg_html, unsafe_allow_html=True)

                        # Aligned badges
                        aligned_html = "**Aligned Joints (Good Form):**<br>"
                        if comparison.aligned_joints:
                            for aj in comparison.aligned_joints:
                                aligned_html += f'<span class="badge-aligned">✓ {aj}</span>'
                        else:
                            aligned_html += "<em>No joints aligned yet. Adjust posture.</em>"
                        aligned_placeholder.markdown(aligned_html, unsafe_allow_html=True)

                        # Correction badges
                        fix_html = "**Joints Needing Adjustment:**<br>"
                        if comparison.correction_joints:
                            for cj in comparison.correction_joints:
                                fix_html += f'<span class="badge-correction">! {cj}</span>'
                        else:
                            fix_html += '<span class="badge-aligned">All evaluated joints aligned!</span>'
                        fix_placeholder.markdown(fix_html, unsafe_allow_html=True)

                    else:
                        live_alert_placeholder.warning("⚠️ No person detected in camera view. Step back into full frame.")
                        score_placeholder.markdown(
                            '<div class="score-banner score-poor">'
                            '<div style="font-size: 1.5rem;">0% — Searching for Subject</div>'
                            '<div>Stand back so your body is visible</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        progress_placeholder.progress(0.0)
                        suggestions_placeholder.info("Waiting for full body visibility in camera...")
                        aligned_placeholder.markdown("**Aligned Joints:** —")
                        fix_placeholder.markdown("**Joints Needing Adjustment:** —")

                    # Handle Snapshot Capture
                    if capture_btn and has_pose and comparison:
                        cap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
                        os.makedirs(cap_dir, exist_ok=True)
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"smartpose_{ts}_{int(comparison.overall_score)}pct.jpg"
                        filepath = os.path.join(cap_dir, filename)

                        # Overlay diagnostic scorecard directly onto saved image
                        saved_img = annotated_frame.copy()
                        cv2.rectangle(saved_img, (0, h - 90), (w, h), (20, 24, 33), -1)
                        cv2.putText(
                            saved_img,
                            f"SMARTPOSE RESULT: {chosen_pose} | Score: {comparison.overall_score}% ({comparison.grade})",
                            (25, h - 50),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            saved_img,
                            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 33 Landmarks Tracked",
                            (25, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (160, 200, 240),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imwrite(filepath, saved_img)

                        st.session_state.session_captures.append(
                            {
                                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                                "pose_name": chosen_pose,
                                "score": comparison.overall_score,
                                "grade": comparison.grade,
                                "filepath": filepath,
                                "image_bytes": cv2.imencode(".jpg", saved_img)[1].tobytes(),
                            }
                        )
                        capture_status_placeholder.success(f"Snapshot saved: `{filename}`! View in Results & History.")

                    time.sleep(0.01)

            finally:
                cap.release()
                camera_placeholder.empty()


# =============================================================================
# PAGE 3: REFERENCE POSE GALLERY
# =============================================================================
elif st.session_state.nav_page == "🖼️ Reference Pose Gallery":
    st.markdown('<div class="main-title">🖼️ Reference Pose Gallery</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore certified target postures with biomechanical target angles and benefits.</div>',
        unsafe_allow_html=True,
    )

    ref_mgr = get_reference_manager()
    poses = ref_mgr.list_predefined_poses()

    for name, p in poses.items():
        with st.container():
            c1, c2, c3 = st.columns([1.5, 2, 1.5])
            with c1:
                img = ref_mgr.get_reference_image(name)
                if img is not None:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown(f"### **{p['name']}**")
                st.caption(f"**Category:** {p['category']} | **Difficulty:** {p['difficulty']}")
                st.write(p["description"])
                st.info(f"**Key Health Benefits:** {p['benefits']}")

                # Launch analysis button for this pose
                st.button(
                    f"🎯 Practice {p['name']}",
                    key=f"btn_{p['id']}",
                    on_click=select_pose_and_navigate,
                    args=(name,),
                )

            with c3:
                st.markdown("**Target Joint Angles:**")
                for j_key, val in p["target_angles"].items():
                    j_name = j_key.replace("_", " ").title()
                    st.markdown(f"- **{j_name}:** `{int(val)}°`")

        st.markdown("---")


# =============================================================================
# PAGE 4: RESULTS & HISTORY
# =============================================================================
elif st.session_state.nav_page == "📊 Results & History":
    st.markdown('<div class="main-title">📊 Session Results & History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Review captured pose snapshots, similarity grades, and download performance cards.</div>',
        unsafe_allow_html=True,
    )

    captures = st.session_state.session_captures

    if not captures:
        st.info(
            "No pose snapshots captured yet in this session.\n\n"
            "Go to **🎯 Live Pose Analysis**, start the camera, and click **'📸 Capture & Save Result Snapshot'** "
            "to record your posture alignment!"
        )
        st.button(
            "Go to Live Analysis",
            on_click=change_page,
            args=("🎯 Live Pose Analysis",),
        )
    else:
        st.markdown(f"### Recorded Snapshots: **{len(captures)}**")

        c_action1, c_action2 = st.columns([1.5, 4])
        with c_action1:
            if st.button("🗑️ Clear Session History"):
                st.session_state.session_captures = []
                st.rerun()

        for idx, cap_item in enumerate(reversed(captures)):
            with st.container():
                st.markdown("---")
                col_img, col_info = st.columns([1.8, 2])
                with col_img:
                    st.image(cap_item["image_bytes"], use_container_width=True)
                with col_info:
                    st.markdown(f"### Snapshot #{len(captures) - idx}: **{cap_item['pose_name']}**")
                    st.markdown(f"**Captured At:** `{cap_item['timestamp']}`")
                    st.markdown(f"**Alignment Score:** `{cap_item['score']}%` — **{cap_item['grade']}**")
                    st.caption(f"Saved to: `{cap_item['filepath']}`")

                    st.download_button(
                        label="⬇️ Download Annotated Snapshot",
                        data=cap_item["image_bytes"],
                        file_name=os.path.basename(cap_item["filepath"]),
                        mime="image/jpeg",
                        key=f"dl_{idx}",
                    )
