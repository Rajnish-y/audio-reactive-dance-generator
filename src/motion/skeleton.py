"""
Phase 3 — Dance Motion: skeleton format + placeholder motion generator.

This defines a simple 2D stick-figure skeleton and two hand-built motion
loops (procedurally generated, not motion-captured). The point is to
prove out the retrieval/rendering pipeline end-to-end with something
simple and fast, before wiring up real motion-capture data (AIST++)
later. The JSON pose format here is what a real dataset would eventually
plug into — swapping the data source later shouldn't require changing
anything downstream that consumes a pose.
"""

import numpy as np

# 15-joint 2D stick figure — a reduced, simplified joint set (similar in
# spirit to common pose-estimation formats like COCO, just smaller).
SKELETON_JOINTS = [
    "head", "neck",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_hand", "r_hand",
    "hip_center", "l_hip", "r_hip", "l_knee", "r_knee", "l_foot", "r_foot",
]

# Bone connections, for drawing lines between joints when rendering.
BONES = [
    ("head", "neck"),
    ("neck", "l_shoulder"), ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_hand"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_hand"),
    ("neck", "hip_center"),
    ("hip_center", "l_hip"), ("hip_center", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_foot"),
    ("r_hip", "r_knee"), ("r_knee", "r_foot"),
]

# A neutral standing pose. All motion is generated as offsets from this.
BASE_POSE = {
    "head": (0.0, 1.70), "neck": (0.0, 1.50),
    "l_shoulder": (-0.25, 1.45), "r_shoulder": (0.25, 1.45),
    "l_elbow": (-0.35, 1.15), "r_elbow": (0.35, 1.15),
    "l_hand": (-0.40, 0.85), "r_hand": (0.40, 0.85),
    "hip_center": (0.0, 0.90),
    "l_hip": (-0.15, 0.90), "r_hip": (0.15, 0.90),
    "l_knee": (-0.18, 0.45), "r_knee": (0.18, 0.45),
    "l_foot": (-0.20, 0.0), "r_foot": (0.20, 0.0),
}

UPPER_BODY = [
    "head", "neck", "l_shoulder", "r_shoulder",
    "hip_center", "l_hip", "r_hip",
]

# The motion types this placeholder generator knows about. A real
# retrieval system (Phase 4) would pick between many more of these,
# selected to match the music's energy/tempo.
AVAILABLE_MOTIONS = ["bounce", "sway"]


def generate_pose(motion_type: str, phase: float, intensity: float = 1.0) -> dict:
    """
    Generate one stick-figure pose for a given motion type at a given
    phase in its loop cycle.

    Args:
        motion_type: one of AVAILABLE_MOTIONS
        phase: position in the loop, in radians (0 to 2*pi is one full cycle)
        intensity: scales the amplitude of the motion (0..1 typical range).
            Lets a quiet section sway subtly and a loud section bounce big,
            using the same motion type — this is the "how much" dimension,
            separate from "which motion" (motion_type).

    Returns:
        dict mapping joint name -> [x, y]
    """
    if motion_type not in AVAILABLE_MOTIONS:
        raise ValueError(f"Unknown motion_type: {motion_type}. Available: {AVAILABLE_MOTIONS}")

    pose = {j: list(BASE_POSE[j]) for j in SKELETON_JOINTS}

    if motion_type == "bounce":
        bob = 0.06 * intensity * abs(np.sin(phase))
        for j in UPPER_BODY:
            pose[j][1] += bob
        arm_swing = 0.08 * intensity * np.sin(phase)
        pose["l_hand"][1] += arm_swing
        pose["r_hand"][1] += arm_swing
        pose["l_elbow"][1] += arm_swing * 0.6
        pose["r_elbow"][1] += arm_swing * 0.6
        knee_bend = 0.03 * intensity * abs(np.sin(phase))
        pose["l_knee"][1] -= knee_bend
        pose["r_knee"][1] -= knee_bend

    elif motion_type == "sway":
        sway = 0.12 * intensity * np.sin(phase)
        for j in UPPER_BODY:
            pose[j][0] += sway
        pose["l_hand"][0] += sway * 1.3
        pose["r_hand"][0] += sway * 1.3
        pose["l_elbow"][0] += sway * 1.15
        pose["r_elbow"][0] += sway * 1.15

    return pose