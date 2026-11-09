"""
SmartPose - Reference Pose Manager
Maintains the catalog of 5 predefined reference poses with target angles,
metadata, and assets, plus supports processing user-uploaded custom poses.
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple


# Predefined Target Reference Angles for the 5 built-in poses
PREDEFINED_POSES = {
    "T-Pose": {
        "id": "t_pose",
        "name": "T-Pose",
        "category": "Calibration & Mobility",
        "description": "Fundamental calibration posture. Stand upright with arms extended straight out horizontally parallel to the ground.",
        "difficulty": "Beginner",
        "benefits": "Improves shoulder stability, spine alignment, and arm endurance.",
        "image_file": "t_pose.png",
        "target_angles": {
            "left_elbow": 180.0,
            "right_elbow": 180.0,
            "left_shoulder": 90.0,
            "right_shoulder": 90.0,
            "left_hip": 180.0,
            "right_hip": 180.0,
            "left_knee": 180.0,
            "right_knee": 180.0,
            "spine_vertical": 0.0,
            "shoulder_tilt": 0.0,
        },
    },
    "Tree Pose": {
        "id": "tree_pose",
        "name": "Tree Pose (Vrksasana)",
        "category": "Balance & Focus",
        "description": "Standing on the left leg, right knee is bent outward with foot resting against inner thigh. Palms pressed at heart center.",
        "difficulty": "Intermediate",
        "benefits": "Builds unilateral leg strength, ankle stability, and mental focus.",
        "image_file": "tree_pose.png",
        "target_angles": {
            "left_elbow": 65.0,
            "right_elbow": 65.0,
            "left_shoulder": 25.0,
            "right_shoulder": 25.0,
            "left_hip": 180.0,
            "right_hip": 130.0,
            "left_knee": 180.0,
            "right_knee": 50.0,
            "spine_vertical": 0.0,
            "shoulder_tilt": 0.0,
        },
    },
    "Warrior Pose": {
        "id": "warrior_pose",
        "name": "Warrior Pose (Virabhadrasana II)",
        "category": "Strength & Endurance",
        "description": "Lunge stance with front knee bent at 90°, back leg straight, and arms extended parallel to floor pointing front and back.",
        "difficulty": "Intermediate",
        "benefits": "Strengthens quadriceps, opens hips, and tones the core.",
        "image_file": "warrior_pose.png",
        "target_angles": {
            "left_elbow": 180.0,
            "right_elbow": 180.0,
            "left_shoulder": 90.0,
            "right_shoulder": 90.0,
            "left_hip": 105.0,
            "right_hip": 165.0,
            "left_knee": 95.0,
            "right_knee": 175.0,
            "spine_vertical": 0.0,
            "shoulder_tilt": 0.0,
        },
    },
    "Standing Pose": {
        "id": "standing_pose",
        "name": "Standing Pose (Neutral Upright)",
        "category": "Postural Alignment",
        "description": "Natural upright anatomical posture. Feet hip-width apart, arms relaxed at sides, shoulders rolled back and relaxed.",
        "difficulty": "Beginner",
        "benefits": "Corrects forward-head slouch, relieves spinal compression, and improves breathing.",
        "image_file": "standing_pose.png",
        "target_angles": {
            "left_elbow": 175.0,
            "right_elbow": 175.0,
            "left_shoulder": 15.0,
            "right_shoulder": 15.0,
            "left_hip": 180.0,
            "right_hip": 180.0,
            "left_knee": 180.0,
            "right_knee": 180.0,
            "spine_vertical": 0.0,
            "shoulder_tilt": 0.0,
        },
    },
    "Mountain Pose": {
        "id": "mountain_pose",
        "name": "Mountain Pose (Tadasana - Extended)",
        "category": "Extension & Grounding",
        "description": "Tall grounded stance with arms reaching straight overhead towards the sky, palms facing inwards, lengthening the entire spine.",
        "difficulty": "Beginner",
        "benefits": "Lengthens the spine, increases shoulder flexibility, and enhances full-body posture awareness.",
        "image_file": "mountain_pose.png",
        "target_angles": {
            "left_elbow": 180.0,
            "right_elbow": 180.0,
            "left_shoulder": 175.0,
            "right_shoulder": 175.0,
            "left_hip": 180.0,
            "right_hip": 180.0,
            "left_knee": 180.0,
            "right_knee": 180.0,
            "spine_vertical": 0.0,
            "shoulder_tilt": 0.0,
        },
    },
}


class ReferencePoseManager:
    """Provides reference pose data, images, and custom uploaded pose ingestion."""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.base_dir, "assets", "reference_poses")
        os.makedirs(self.assets_dir, exist_ok=True)

    def list_predefined_poses(self) -> Dict[str, dict]:
        """Returns the dictionary of predefined poses."""
        return PREDEFINED_POSES

    def get_pose_by_name(self, name: str) -> Optional[dict]:
        """Fetches pose specification by name."""
        return PREDEFINED_POSES.get(name)

    def get_reference_image(self, name: str) -> Optional[np.ndarray]:
        """Loads and returns the BGR image array for a predefined pose."""
        pose_info = self.get_pose_by_name(name)
        if not pose_info:
            return None
        img_path = os.path.join(self.assets_dir, pose_info["image_file"])
        if os.path.exists(img_path):
            return cv2.imread(img_path)
        return None

    def process_custom_upload(self, detector, uploaded_bytes: bytes) -> Tuple[bool, Optional[np.ndarray], Optional[Dict[str, float]], str]:
        """
        Processes an uploaded image file:
        - Decodes image
        - Detects human pose using detector
        - Extracts target joint angles
        - Draws annotated skeleton
        Returns: (success, annotated_image_bgr, target_angles, message)
        """
        # Decode image from byte buffer
        nparr = np.frombuffer(uploaded_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, None, None, "Invalid image file format. Please upload a valid JPG or PNG."

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.process_frame(rgb)

        if not result or not result.pose_landmarks:
            return False, img, None, "No human pose detected in uploaded image. Please ensure full body is clearly visible."

        landmarks = result.pose_landmarks[0]
        h, w = img.shape[:2]
        lm_dict = detector.extract_landmark_dict(landmarks, w, h)
        target_angles = detector.extract_joint_angles(lm_dict)

        annotated_img = img.copy()
        detector.draw_annotated_skeleton(annotated_img, landmarks, angles=target_angles, draw_angles=True)

        return True, annotated_img, target_angles, f"Successfully extracted {len(target_angles)} joint angles from custom pose!"

