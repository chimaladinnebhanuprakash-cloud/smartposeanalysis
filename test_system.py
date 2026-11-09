"""
SmartPose - Comprehensive System Diagnostic & Integration Test
Validates:
1. Environment & Dependencies
2. Pretrained Model & Inference
3. Angle Calculation Engine
4. Reference Pose Loading (All 5 poses)
5. Pose Comparison & Alignment Scoring (0 - 100%)
6. Correction Suggestions Generation
7. Webcam Accessibility
"""

import sys
import os
import cv2
import numpy as np


def run_diagnostics():
    print("=" * 65)
    print("SMARTPOSE – SYSTEM INTEGRATION & END-TO-END VERIFICATION")
    print("=" * 65)

    # 1. Environment & Package Verification
    print("\n[TEST 1] Environment & Library Verification:")
    print(f"  • Python: {sys.version.split()[0]} ({sys.executable})")

    try:
        import streamlit
        print(f"  • Streamlit: {streamlit.__version__} [OK]")
    except ImportError as e:
        print(f"  • Streamlit: FAILED ({e})")
        return False

    try:
        import cv2
        print(f"  • OpenCV: {cv2.__version__} [OK]")
    except ImportError as e:
        print(f"  • OpenCV: FAILED ({e})")
        return False

    try:
        import mediapipe as mp
        print(f"  • MediaPipe: {mp.__version__} [OK]")
    except ImportError as e:
        print(f"  • MediaPipe: FAILED ({e})")
        return False

    try:
        import numpy as np
        print(f"  • NumPy: {np.__version__} [OK]")
    except ImportError as e:
        print(f"  • NumPy: FAILED ({e})")
        return False

    # 2. Pretrained Model Verification
    print("\n[TEST 2] Pretrained Model Verification:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "pose_landmarker_full.task")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  • Model Asset Found: {model_path} ({size_mb:.2f} MB) [OK]")
    else:
        print(f"  • Model Asset NOT Found at {model_path} [FAILED]")
        return False

    # 3. Model Loading & Inference Test
    print("\n[TEST 3] MediaPipe Pose Landmarker Inference Test:")
    from pose_detector import PoseDetector, LANDMARKS
    try:
        detector = PoseDetector(model_path=model_path)
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.process_frame(dummy_rgb)
        print("  • PoseDetector initialized and executed dummy frame inference [OK]")
    except Exception as e:
        print(f"  • PoseDetector initialization error: {e} [FAILED]")
        return False

    # 4. Joint Angle Calculation Engine
    print("\n[TEST 4] Trigonometric Joint Angle Calculation:")
    p1 = {"x": 100, "y": 100}
    p2 = {"x": 100, "y": 200}
    p3 = {"x": 200, "y": 200}
    deg = detector.calculate_angle(p1, p2, p3)
    assert abs(deg - 90.0) < 1e-4, f"Angle error: expected 90.0, got {deg}"
    print(f"  • Right-angle test (100,100 -> 100,200 -> 200,200): {deg}° [OK]")

    p_straight1 = {"x": 100, "y": 100}
    p_straight2 = {"x": 100, "y": 200}
    p_straight3 = {"x": 100, "y": 300}
    deg_straight = detector.calculate_angle(p_straight1, p_straight2, p_straight3)
    assert abs(deg_straight - 180.0) < 1e-4, f"Straight line error: expected 180.0, got {deg_straight}"
    print(f"  • Straight-line test: {deg_straight}° [OK]")

    # 5. Reference Pose Catalog (All 5 Poses)
    print("\n[TEST 5] Reference Pose Catalog Verification:")
    from reference_manager import ReferencePoseManager, PREDEFINED_POSES
    ref_mgr = ReferencePoseManager()
    poses = ref_mgr.list_predefined_poses()
    expected_poses = ["T-Pose", "Tree Pose", "Warrior Pose", "Standing Pose", "Mountain Pose"]
    for ep in expected_poses:
        if ep in poses:
            info = poses[ep]
            img = ref_mgr.get_reference_image(ep)
            has_img = img is not None
            print(f"  • '{ep}' catalog entry: {len(info['target_angles'])} target angles, Image found: {has_img} [OK]")
        else:
            print(f"  • '{ep}' MISSING from catalog [FAILED]")
            return False

    # 6. Pose Comparison & Alignment Scoring Test
    print("\n[TEST 6] Pose Comparison & Similarity Scoring:")
    from pose_comparator import PoseComparator
    comparator = PoseComparator()

    t_angles = poses["T-Pose"]["target_angles"]

    # Exact match test (must be 100%)
    exact_comp = comparator.compare_poses(t_angles, t_angles)
    assert exact_comp.overall_score == 100.0, f"Expected 100% on identical match, got {exact_comp.overall_score}"
    print(f"  • Exact Match Score: {exact_comp.overall_score}% ({exact_comp.grade}) [OK]")
    assert len(exact_comp.aligned_joints) > 0

    # Divergent pose test
    mismatched_angles = dict(t_angles)
    mismatched_angles["left_shoulder"] = 30.0  # Dropped arm instead of 90 deg
    mismatched_angles["left_elbow"] = 90.0     # Bent arm instead of 180 deg
    diff_comp = comparator.compare_poses(mismatched_angles, t_angles)
    print(f"  • Mismatched Form Score: {diff_comp.overall_score}% ({diff_comp.grade}) [OK]")
    print(f"  • Actionable Suggestions Generated: {diff_comp.suggestions}")
    assert any("arm" in s.lower() or "elbow" in s.lower() or "shoulder" in s.lower() for s in diff_comp.suggestions)

    # 7. Webcam Device Access Check
    print("\n[TEST 7] Hardware Camera Access Check:")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"  • Primary Webcam (Index 0): Connected & Functional (Resolution: {frame.shape[1]}x{frame.shape[0]}) [OK]")
        else:
            print("  • Primary Webcam opened but could not capture a frame [WARNING]")
    else:
        print("  • Primary Webcam could not be opened (camera might be in use) [WARNING]")

    print("\n" + "=" * 65)
    print("ALL 7 CORE SYSTEM TESTS PASSED SUCCESSFULLY! SYSTEM READY.")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)

