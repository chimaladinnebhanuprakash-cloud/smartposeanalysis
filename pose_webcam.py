"""
Real-time MediaPipe Pose Detection with Webcam
Powered by Google MediaPipe Tasks API (Vision PoseLandmarker)
"""

import os
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, PoseLandmarksConnections


def get_model_path():
    """Resolve the path to the pretrained pose landmarker model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pretrained model file not found at: {model_path}\n"
            "Please ensure pose_landmarker_full.task is present in the project directory."
        )
    return model_path


def run_webcam_pose(camera_index: int = 0):
    """Run real-time pose detection on the webcam feed."""
    model_path = get_model_path()
    print(f"Loading pretrained pose model from: {model_path}")

    # Configure PoseLandmarker
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        return

    print("Camera initialized successfully. Press 'q' or 'ESC' in the window to exit.")

    prev_time = time.time()
    fps = 0.0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Warning: Empty frame received from webcam. Skipping...")
                continue

            # Flip frame horizontally for natural selfie-view mirror effect
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform pose detection
            results = landmarker.detect(mp_image)

            # Draw pose landmarks and connections
            if results.pose_landmarks:
                for pose_landmarks in results.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=pose_landmarks,
                        connections=PoseLandmarksConnections.POSE_LANDMARKS,
                    )

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # Overlay FPS and status information
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            status_text = (
                f"Pose Detected: {len(results.pose_landmarks)}"
                if results.pose_landmarks
                else "No Pose Detected"
            )
            cv2.putText(
                frame,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255) if results.pose_landmarks else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # Display output
            cv2.imshow("MediaPipe Pose Detection", frame)

            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("Exit requested by user.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stream stopped and resources released cleanly.")


if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam_pose(camera_idx)

