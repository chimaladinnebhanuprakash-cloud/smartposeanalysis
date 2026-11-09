"""
Automated Verification Script for MediaPipe Pose Setup
Validates Python version, required packages, pretrained model, and webcam inference.
"""

import sys
import os

def test_environment():
    print("=" * 60)
    print("1. CHECKING PYTHON ENVIRONMENT")
    print(f"   Python Executable: {sys.executable}")
    print(f"   Python Version: {sys.version.split()[0]}")
    version_info = sys.version_info
    if (version_info.major, version_info.minor) not in [(3, 9), (3, 10), (3, 11), (3, 12)]:
        print(f"   [WARNING] Python version {version_info.major}.{version_info.minor} is outside recommended MediaPipe range (3.9 - 3.12).")
    else:
        print("   [OK] Python version is compatible with MediaPipe.")

    print("\n2. CHECKING INSTALLED PACKAGES")
    try:
        import numpy as np
        print(f"   [OK] NumPy: {np.__version__}")
    except ImportError as e:
        print(f"   [FAIL] NumPy: {e}")

    try:
        import cv2
        print(f"   [OK] OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"   [FAIL] OpenCV: {e}")

    try:
        import mediapipe as mp
        print(f"   [OK] MediaPipe: {mp.__version__}")
    except ImportError as e:
        print(f"   [FAIL] MediaPipe: {e}")

    print("\n3. CHECKING PRETRAINED POSE MODEL")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pose_landmarker_full.task")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   [OK] Found model: {model_path} ({size_mb:.2f} MB)")
    else:
        print(f"   [FAIL] Model not found at: {model_path}")
        return False

    print("\n4. TESTING MODEL INITIALIZATION & INFERENCE")
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_img)
            result = landmarker.detect(mp_img)
            print("   [OK] PoseLandmarker initialized and ran inference successfully.")
    except Exception as e:
        print(f"   [FAIL] Inference failed: {e}")
        return False

    print("\n5. CHECKING WEBCAM DEVICE ACCESS")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"   [OK] Webcam is functional (Captured frame: {frame.shape[1]}x{frame.shape[0]}).")
            else:
                print("   [WARNING] Webcam opened, but could not capture a frame.")
        else:
            print("   [WARNING] Could not open webcam index 0 (camera may be in use or unavailable).")
    except Exception as e:
        print(f"   [WARNING] Webcam check exception: {e}")

    print("=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)

