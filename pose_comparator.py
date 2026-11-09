"""
SmartPose - Pose Comparison & Alignment Scoring Engine
Compares live pose landmarks and angles with reference pose targets,
computes a mathematically sound similarity score (0 - 100%),
and provides real-time actionable correction suggestions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


JOINT_LABELS = {
    "left_elbow": "Left Elbow",
    "right_elbow": "Right Elbow",
    "left_shoulder": "Left Shoulder",
    "right_shoulder": "Right Shoulder",
    "left_hip": "Left Hip",
    "right_hip": "Right Hip",
    "left_knee": "Left Knee",
    "right_knee": "Right Knee",
    "spine_vertical": "Spine Verticality",
    "shoulder_tilt": "Shoulder Level",
}


@dataclass
class JointComparison:
    joint_key: str
    label: str
    live_angle: float
    ref_angle: float
    diff: float
    score: float
    status: str  # 'aligned', 'warning', 'misaligned'
    suggestion: str


@dataclass
class ComparisonResult:
    overall_score: float
    grade: str
    feedback: str
    joint_results: Dict[str, JointComparison]
    aligned_joints: List[str]
    correction_joints: List[str]
    suggestions: List[str]


class PoseComparator:
    """Analyzes and compares joint angles between live and reference poses."""

    def __init__(self, tolerance_deg: float = 15.0, warning_deg: float = 30.0):
        self.tolerance_deg = tolerance_deg
        self.warning_deg = warning_deg

    def generate_joint_suggestion(self, joint_key: str, live_val: float, ref_val: float, diff: float) -> str:
        """Produces precise, human-readable correction advice for an individual joint."""
        if diff <= self.tolerance_deg:
            return f"{JOINT_LABELS.get(joint_key, joint_key)} is well aligned."

        delta = live_val - ref_val

        # Elbows (Extension vs Flexion)
        if joint_key == "left_elbow":
            if delta < 0:
                return "Straighten your left arm (extend elbow outward)."
            else:
                return "Bend your left elbow more."

        if joint_key == "right_elbow":
            if delta < 0:
                return "Straighten your right arm (extend elbow outward)."
            else:
                return "Bend your right elbow more."

        # Shoulders (Elevation / Abduction vs Lowering)
        if joint_key == "left_shoulder":
            if delta < 0:
                return "Raise your left arm higher to match the reference angle."
            else:
                return "Lower your left arm slightly."

        if joint_key == "right_shoulder":
            if delta < 0:
                return "Raise your right arm higher to match the reference angle."
            else:
                return "Lower your right arm slightly."

        # Hips (Torso-Thigh Flexion)
        if joint_key in ["left_hip", "right_hip"]:
            side = "left" if "left" in joint_key else "right"
            if delta < 0:
                return f"Extend your {side} hip and stand taller."
            else:
                return f"Hinge your {side} hip closer to reference."

        # Knees (Straight vs Bent)
        if joint_key == "left_knee":
            if delta < 0:
                return "Straighten your left knee."
            else:
                return "Bend your left knee more."

        if joint_key == "right_knee":
            if delta < 0:
                return "Straighten your right knee."
            else:
                return "Bend your right knee more."

        # Spine Verticality
        if joint_key == "spine_vertical":
            return "Keep your spine upright and engage your core."

        # Shoulder Tilt
        if joint_key == "shoulder_tilt":
            return "Keep your shoulders horizontal and square."

        return f"Adjust your {JOINT_LABELS.get(joint_key, joint_key)} position."

    def compare_poses(self, live_angles: Dict[str, float], ref_angles: Dict[str, float]) -> ComparisonResult:
        """
        Compares live angles with reference angles.
        Returns a rich ComparisonResult with percentage similarity and action items.
        """
        if not live_angles or not ref_angles:
            return ComparisonResult(
                overall_score=0.0,
                grade="No Subject",
                feedback="Stand in front of the camera with your full body visible.",
                joint_results={},
                aligned_joints=[],
                correction_joints=[],
                suggestions=["Step back into the camera frame so your body landmarks can be detected."],
            )

        common_keys = [k for k in ref_angles.keys() if k in live_angles and k in JOINT_LABELS]
        if not common_keys:
            return ComparisonResult(
                overall_score=0.0,
                grade="No Matching Keypoints",
                feedback="Unable to evaluate posture landmarks.",
                joint_results={},
                aligned_joints=[],
                correction_joints=[],
                suggestions=["Ensure key joints are not occluded."],
            )

        joint_results = {}
        aligned_joints = []
        correction_joints = []
        suggestions = []
        joint_scores = []

        # Relative importance weights for key joints
        weights = {
            "left_elbow": 1.0,
            "right_elbow": 1.0,
            "left_shoulder": 1.2,
            "right_shoulder": 1.2,
            "left_knee": 1.1,
            "right_knee": 1.1,
            "left_hip": 1.0,
            "right_hip": 1.0,
            "spine_vertical": 1.0,
            "shoulder_tilt": 0.8,
        }

        total_weight = 0.0
        weighted_score_sum = 0.0

        for key in common_keys:
            live_val = live_angles[key]
            ref_val = ref_angles[key]
            diff = abs(live_val - ref_val)

            # Score per joint: 100 at 0 diff, 0 at 50+ deg diff
            joint_score = max(0.0, min(100.0, 100.0 - (diff / 45.0) * 100.0))
            w = weights.get(key, 1.0)
            weighted_score_sum += joint_score * w
            total_weight += w

            # Classification
            if diff <= self.tolerance_deg:
                status = "aligned"
                aligned_joints.append(JOINT_LABELS[key])
            elif diff <= self.warning_deg:
                status = "warning"
                correction_joints.append(JOINT_LABELS[key])
                sugg = self.generate_joint_suggestion(key, live_val, ref_val, diff)
                suggestions.append(sugg)
            else:
                status = "misaligned"
                correction_joints.append(JOINT_LABELS[key])
                sugg = self.generate_joint_suggestion(key, live_val, ref_val, diff)
                suggestions.append(sugg)

            joint_results[key] = JointComparison(
                joint_key=key,
                label=JOINT_LABELS[key],
                live_angle=live_val,
                ref_angle=ref_val,
                diff=round(diff, 1),
                score=round(joint_score, 1),
                status=status,
                suggestion=self.generate_joint_suggestion(key, live_val, ref_val, diff),
            )

        overall_score = round(weighted_score_sum / max(total_weight, 1e-6), 1)

        # Determine overall grade & feedback
        if overall_score >= 90.0:
            grade = "🌟 Excellent Alignment"
            feedback = "Superb posture! Your joint angles match the reference model with high accuracy."
        elif overall_score >= 75.0:
            grade = "✅ Good Alignment"
            feedback = "Solid posture! Follow the minor correction tips below to achieve perfection."
        elif overall_score >= 55.0:
            grade = "⚠️ Moderate Alignment"
            feedback = "Noticeable deviations detected. Adjust your limb positions to align with the reference."
        else:
            grade = "❌ Poor Alignment"
            feedback = "Significant postural mismatch. Look at the reference pose and reset your body position."

        # If aligned, provide positive reinforcement
        if not suggestions:
            suggestions.append("Your pose matches the reference target! Hold this position.")

        return ComparisonResult(
            overall_score=overall_score,
            grade=grade,
            feedback=feedback,
            joint_results=joint_results,
            aligned_joints=aligned_joints,
            correction_joints=correction_joints,
            suggestions=suggestions[:4],  # Top actionable suggestions
        )

