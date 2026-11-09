"""
SmartPose - Pose Detection and Landmark Processing Engine
Powered by MediaPipe Tasks API (Vision PoseLandmarker).
Provides 33-landmark extraction, joint angle computation, and custom visual overlays.
"""

import os
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, PoseLandmarksConnections


# Full MediaPipe Pose 33 Landmark Indices
LANDMARKS = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9,
    "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# Major Skeleton Connections for Custom Highlighting
SKELETON_CONNECTIONS = [
    (11, 12),  # Left Shoulder to Right Shoulder
    (11, 13),  # Left Shoulder to Left Elbow
    (13, 15),  # Left Elbow to Left Wrist
    (12, 14),  # Right Shoulder to Right Elbow
    (14, 16),  # Right Elbow to Right Wrist
    (11, 23),  # Left Shoulder to Left Hip
    (12, 24),  # Right Shoulder to Right Hip
    (23, 24),  # Left Hip to Right Hip
    (23, 25),  # Left Hip to Left Knee
    (25, 27),  # Left Knee to Left Ankle
    (24, 26),  # Right Hip to Right Knee
    (26, 28),  # Right Knee to Right Ankle
    (27, 29),  # Left Ankle to Left Heel
    (27, 31),  # Left Ankle to Left Foot Index
    (28, 30),  # Right Ankle to Right Heel
    (28, 32),  # Right Ankle to Right Foot Index
]


class PoseDetector:
    """Wrapper around MediaPipe PoseLandmarker for landmark and angle tracking."""

    def __init__(
        self,
        model_path: str = None,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "pose_landmarker_full.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model file not found: {model_path}")

        self.model_path = model_path
        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, rgb_image: np.ndarray):
        """Processes an RGB image and returns the detection result."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect(mp_image)

    @staticmethod
    def extract_landmark_dict(landmarks, width: int, height: int):
        """
        Converts NormalizedLandmarks list to a dictionary of coordinates indexed by name.
        """
        lm_dict = {}
        for name, idx in LANDMARKS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                lm_dict[name] = {
                    "x": int(lm.x * width),
                    "y": int(lm.y * height),
                    "norm_x": float(lm.x),
                    "norm_y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 1.0)),
                    "presence": float(getattr(lm, "presence", 1.0)),
                }
        return lm_dict

    @staticmethod
    def calculate_angle(pt1, pt2, pt3) -> float:
        """
        Calculates the angle in degrees at vertex pt2 between rays pt2->pt1 and pt2->pt3.
        Points can be dicts with 'x' and 'y' or tuples/lists (x, y).
        Returns angle in [0.0, 180.0].
        """
        p1 = (pt1["x"], pt1["y"]) if isinstance(pt1, dict) else pt1
        p2 = (pt2["x"], pt2["y"]) if isinstance(pt2, dict) else pt2
        p3 = (pt3["x"], pt3["y"]) if isinstance(pt3, dict) else pt3

        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0

        cos_angle = dot / (mag1 * mag2)
        # Numerical clamp
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        return round(math.degrees(angle_rad), 1)

    def extract_joint_angles(self, lm_dict: dict) -> dict:
        """
        Calculates all anatomical key joint angles from landmark dictionary.
        Returns dict with angle values in degrees.
        """
        angles = {}

        # 1. Left Elbow (Shoulder - Elbow - Wrist)
        if all(k in lm_dict for k in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]):
            angles["left_elbow"] = self.calculate_angle(
                lm_dict["LEFT_SHOULDER"], lm_dict["LEFT_ELBOW"], lm_dict["LEFT_WRIST"]
            )

        # 2. Right Elbow (Shoulder - Elbow - Wrist)
        if all(k in lm_dict for k in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]):
            angles["right_elbow"] = self.calculate_angle(
                lm_dict["RIGHT_SHOULDER"], lm_dict["RIGHT_ELBOW"], lm_dict["RIGHT_WRIST"]
            )

        # 3. Left Shoulder (Elbow - Shoulder - Hip)
        if all(k in lm_dict for k in ["LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"]):
            angles["left_shoulder"] = self.calculate_angle(
                lm_dict["LEFT_ELBOW"], lm_dict["LEFT_SHOULDER"], lm_dict["LEFT_HIP"]
            )

        # 4. Right Shoulder (Elbow - Shoulder - Hip)
        if all(k in lm_dict for k in ["RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"]):
            angles["right_shoulder"] = self.calculate_angle(
                lm_dict["RIGHT_ELBOW"], lm_dict["RIGHT_SHOULDER"], lm_dict["RIGHT_HIP"]
            )

        # 5. Left Hip (Shoulder - Hip - Knee)
        if all(k in lm_dict for k in ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"]):
            angles["left_hip"] = self.calculate_angle(
                lm_dict["LEFT_SHOULDER"], lm_dict["LEFT_HIP"], lm_dict["LEFT_KNEE"]
            )

        # 6. Right Hip (Shoulder - Hip - Knee)
        if all(k in lm_dict for k in ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"]):
            angles["right_hip"] = self.calculate_angle(
                lm_dict["RIGHT_SHOULDER"], lm_dict["RIGHT_HIP"], lm_dict["RIGHT_KNEE"]
            )

        # 7. Left Knee (Hip - Knee - Ankle)
        if all(k in lm_dict for k in ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"]):
            angles["left_knee"] = self.calculate_angle(
                lm_dict["LEFT_HIP"], lm_dict["LEFT_KNEE"], lm_dict["LEFT_ANKLE"]
            )

        # 8. Right Knee (Hip - Knee - Ankle)
        if all(k in lm_dict for k in ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]):
            angles["right_knee"] = self.calculate_angle(
                lm_dict["RIGHT_HIP"], lm_dict["RIGHT_KNEE"], lm_dict["RIGHT_ANKLE"]
            )

        # 9. Torso / Spine Vertical Alignment
        # Angle of spine line (mid-shoulder to mid-hip) relative to true vertical (0 deg = straight)
        if all(k in lm_dict for k in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]):
            mid_sh_x = (lm_dict["LEFT_SHOULDER"]["x"] + lm_dict["RIGHT_SHOULDER"]["x"]) / 2.0
            mid_sh_y = (lm_dict["LEFT_SHOULDER"]["y"] + lm_dict["RIGHT_SHOULDER"]["y"]) / 2.0
            mid_hip_x = (lm_dict["LEFT_HIP"]["x"] + lm_dict["RIGHT_HIP"]["x"]) / 2.0
            mid_hip_y = (lm_dict["LEFT_HIP"]["y"] + lm_dict["RIGHT_HIP"]["y"]) / 2.0

            dx = mid_sh_x - mid_hip_x
            dy = mid_hip_y - mid_sh_y  # In image coordinates, y increases downwards
            angle_from_vertical = abs(math.degrees(math.atan2(dx, dy + 1e-6)))
            angles["spine_vertical"] = round(angle_from_vertical, 1)

        # 10. Shoulder Horizontal Level
        if all(k in lm_dict for k in ["LEFT_SHOULDER", "RIGHT_SHOULDER"]):
            sh_dx = lm_dict["RIGHT_SHOULDER"]["x"] - lm_dict["LEFT_SHOULDER"]["x"]
            sh_dy = lm_dict["RIGHT_SHOULDER"]["y"] - lm_dict["LEFT_SHOULDER"]["y"]
            shoulder_tilt = abs(math.degrees(math.atan2(sh_dy, sh_dx + 1e-6)))
            angles["shoulder_tilt"] = round(shoulder_tilt, 1)

        return angles

    def draw_annotated_skeleton(
        self,
        image: np.ndarray,
        landmarks,
        joint_status: dict = None,
        draw_angles: bool = True,
        angles: dict = None,
    ) -> np.ndarray:
        """
        Draws skeleton connections and color-coded joint nodes.
        joint_status: optional dict mapping joint names to status ('aligned', 'warning', 'misaligned')
        """
        h, w = image.shape[:2]
        lm_dict = self.extract_landmark_dict(landmarks, w, h)

        # Color palette (BGR)
        COLOR_ALIGNED = (0, 230, 115)     # Vibrant Green
        COLOR_WARNING = (0, 204, 255)     # Amber / Yellow
        COLOR_MISALIGNED = (60, 60, 255)  # Bright Red
        COLOR_DEFAULT = (240, 240, 240)   # Clean White
        COLOR_BONE = (200, 215, 235)      # Soft Cyan/Blue

        # Mapping joint names to MediaPipe landmark vertex index
        JOINT_TO_LM = {
            "left_elbow": "LEFT_ELBOW",
            "right_elbow": "RIGHT_ELBOW",
            "left_shoulder": "LEFT_SHOULDER",
            "right_shoulder": "RIGHT_SHOULDER",
            "left_hip": "LEFT_HIP",
            "right_hip": "RIGHT_HIP",
            "left_knee": "LEFT_KNEE",
            "right_knee": "RIGHT_KNEE",
        }

        # 1. Draw Skeleton Lines
        for idx1, idx2 in SKELETON_CONNECTIONS:
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                p1 = landmarks[idx1]
                p2 = landmarks[idx2]
                if getattr(p1, "visibility", 1.0) > 0.4 and getattr(p2, "visibility", 1.0) > 0.4:
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    # Outer subtle shadow
                    cv2.line(image, pt1, pt2, (40, 40, 40), 6, cv2.LINE_AA)
                    # Main bone
                    cv2.line(image, pt1, pt2, COLOR_BONE, 3, cv2.LINE_AA)

        # 2. Draw Nodes with color coding
        for joint_key, lm_key in JOINT_TO_LM.items():
            if lm_key in lm_dict:
                info = lm_dict[lm_key]
                if info["visibility"] > 0.4:
                    pt = (info["x"], info["y"])
                    status = (joint_status or {}).get(joint_key, "aligned")
                    
                    if status == "aligned":
                        color = COLOR_ALIGNED
                    elif status == "warning":
                        color = COLOR_WARNING
                    else:
                        color = COLOR_MISALIGNED

                    # Outer glow ring
                    cv2.circle(image, pt, 11, color, 2, cv2.LINE_AA)
                    # Solid center node
                    cv2.circle(image, pt, 7, color, -1, cv2.LINE_AA)
                    cv2.circle(image, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

                    # Draw angle text label next to joint if enabled
                    if draw_angles and angles and joint_key in angles:
                        val = angles[joint_key]
                        txt = f"{int(val)}°"
                        cv2.putText(
                            image,
                            txt,
                            (pt[0] + 12, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image,
                            txt,
                            (pt[0] + 12, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

        # 3. Draw Remaining Landmarks (face, feet, hands) subtly
        for lm in landmarks:
            if getattr(lm, "visibility", 1.0) > 0.5:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(image, pt, 4, COLOR_DEFAULT, -1, cv2.LINE_AA)

        return image

    def close(self):
        """Releases the underlying landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
