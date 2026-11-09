"""
Generates clean, aesthetic reference pose diagrams for the 5 predefined poses:
1. T-Pose
2. Tree Pose (Vrksasana)
3. Warrior Pose (Virabhadrasana II)
4. Standing Pose
5. Mountain Pose (Tadasana)
"""

import os
import math
import cv2
import numpy as np


def create_pose_card(title: str, subtitle: str, keypoints: dict, filename: str):
    """
    Renders an anatomical human pose silhouette on a modern studio background.
    keypoints: dict mapping body part names to (x, y) coordinates on a 600x750 canvas.
    """
    w, h = 600, 750
    # Create dark modern studio gradient background
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        # Subtle vertical gradient from dark blue-slate to deep indigo
        r = int(18 + 15 * ratio)
        g = int(24 + 18 * ratio)
        b = int(38 + 25 * ratio)
        img[y, :] = (b, g, r)

    # Grid / Studio Floor perspective lines
    floor_y = 660
    cv2.line(img, (0, floor_y), (w, floor_y), (60, 75, 105), 2)
    for x in range(50, w, 100):
        cv2.line(img, (x, floor_y), (int(w / 2 + (x - w / 2) * 1.8), h), (45, 55, 80), 1)

    # Header Card overlay
    cv2.rectangle(img, (20, 20), (w - 20, 95), (32, 40, 60), -1)
    cv2.rectangle(img, (20, 20), (w - 20, 95), (70, 90, 130), 1)
    cv2.putText(img, title, (40, 58), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (42, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 190, 255), 1, cv2.LINE_AA)

    # Define skeleton connection pairs
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    # Draw thicker silhouette "limbs"
    for p1_name, p2_name in connections:
        if p1_name in keypoints and p2_name in keypoints:
            pt1 = keypoints[p1_name]
            pt2 = keypoints[p2_name]
            # Outer glow
            cv2.line(img, pt1, pt2, (100, 160, 240), 12, cv2.LINE_AA)
            # Inner bone
            cv2.line(img, pt1, pt2, (230, 240, 255), 5, cv2.LINE_AA)

    # Draw Torso fill polygon
    if all(k in keypoints for k in ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]):
        pts = np.array([
            keypoints["left_shoulder"],
            keypoints["right_shoulder"],
            keypoints["right_hip"],
            keypoints["left_hip"],
        ], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (80, 130, 210))
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Head and Neck
    if "head" in keypoints:
        head_pos = keypoints["head"]
        cv2.circle(img, head_pos, 38, (100, 160, 240), -1, cv2.LINE_AA)
        cv2.circle(img, head_pos, 32, (235, 245, 255), -1, cv2.LINE_AA)
        if "mid_shoulder" in keypoints:
            cv2.line(img, head_pos, keypoints["mid_shoulder"], (230, 240, 255), 6, cv2.LINE_AA)

    # Draw joint nodes
    for name, pt in keypoints.items():
        if name != "head":
            # Outer cyan glow
            cv2.circle(img, pt, 10, (0, 210, 255), -1, cv2.LINE_AA)
            # White center
            cv2.circle(img, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Footer banner
    cv2.putText(
        img,
        "SMARTPOSE REFERENCE TARGET",
        (180, 725),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (140, 160, 190),
        1,
        cv2.LINE_AA,
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "reference_poses")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")


def generate_all():
    # 1. T-Pose
    t_pose = {
        "head": (300, 160),
        "mid_shoulder": (300, 210),
        "left_shoulder": (240, 220),
        "right_shoulder": (360, 220),
        "left_elbow": (130, 220),
        "left_wrist": (40, 220),
        "right_elbow": (470, 220),
        "right_wrist": (560, 220),
        "left_hip": (260, 390),
        "right_hip": (340, 390),
        "left_knee": (260, 520),
        "left_ankle": (260, 645),
        "right_knee": (340, 520),
        "right_ankle": (340, 645),
    }
    create_pose_card("T-Pose", "Arms Horizontal, Body Straight & Upright", t_pose, "t_pose.png")

    # 2. Tree Pose (Vrksasana)
    tree_pose = {
        "head": (300, 150),
        "mid_shoulder": (300, 200),
        "left_shoulder": (250, 215),
        "right_shoulder": (350, 215),
        # Hands together in Anjali Mudra at chest center
        "left_elbow": (210, 270),
        "left_wrist": (295, 270),
        "right_elbow": (390, 270),
        "right_wrist": (305, 270),
        "left_hip": (265, 385),
        "right_hip": (335, 385),
        # Left leg standing vertical
        "left_knee": (270, 520),
        "left_ankle": (270, 645),
        # Right leg bent outward with foot touching inner left knee
        "right_knee": (430, 460),
        "right_ankle": (290, 500),
    }
    create_pose_card("Tree Pose", "Single-Leg Balance with Hands in Prayer", tree_pose, "tree_pose.png")

    # 3. Warrior Pose (Virabhadrasana II)
    warrior_pose = {
        "head": (270, 190),
        "mid_shoulder": (270, 240),
        "left_shoulder": (215, 250),
        "right_shoulder": (325, 250),
        # Arms horizontally extended
        "left_elbow": (120, 250),
        "left_wrist": (35, 250),
        "right_elbow": (420, 250),
        "right_wrist": (530, 250),
        "left_hip": (240, 410),
        "right_hip": (310, 410),
        # Left leg bent forward in 90-degree lunge
        "left_knee": (170, 500),
        "left_ankle": (170, 645),
        # Right leg extended straight back
        "right_knee": (410, 530),
        "right_ankle": (500, 645),
    }
    create_pose_card("Warrior Pose", "Deep Lunge with Extended Parallel Arms", warrior_pose, "warrior_pose.png")

    # 4. Standing Pose (Neutral / Upright)
    standing_pose = {
        "head": (300, 160),
        "mid_shoulder": (300, 210),
        "left_shoulder": (245, 225),
        "right_shoulder": (355, 225),
        # Arms relaxed along the body
        "left_elbow": (225, 335),
        "left_wrist": (215, 440),
        "right_elbow": (375, 335),
        "right_wrist": (385, 440),
        "left_hip": (260, 390),
        "right_hip": (340, 390),
        "left_knee": (265, 520),
        "left_ankle": (265, 645),
        "right_knee": (335, 520),
        "right_ankle": (335, 645),
    }
    create_pose_card("Standing Pose", "Neutral Posture, Arms Natural by Side", standing_pose, "standing_pose.png")

    # 5. Mountain Pose (Tadasana)
    mountain_pose = {
        "head": (300, 210),
        "mid_shoulder": (300, 260),
        "left_shoulder": (245, 270),
        "right_shoulder": (355, 270),
        # Arms extended vertically overhead
        "left_elbow": (235, 160),
        "left_wrist": (230, 55),
        "right_elbow": (365, 160),
        "right_wrist": (370, 55),
        "left_hip": (260, 430),
        "right_hip": (340, 430),
        "left_knee": (265, 545),
        "left_ankle": (265, 650),
        "right_knee": (335, 545),
        "right_ankle": (335, 650),
    }
    create_pose_card("Mountain Pose", "Grounded Stance with Arms Reaching Overhead", mountain_pose, "mountain_pose.png")


if __name__ == "__main__":
    generate_all()

