"""
AIST++ Adapter: convert real AIST++ motion-capture keypoints into this
project's skeleton pose format (see skeleton.py).

AIST++ provides 3D keypoints in standard COCO format (17 joints) per
frame, shape (num_frames, 17, 3) — confirmed directly from Google's
aistplusplus_api loader.py (AISTDataset.load_keypoint3d). This adapter
maps those 17 COCO joints onto our simplified 15-joint skeleton, so
real motion-capture data can be dropped into the existing pipeline
(retrieval.py, rendering, everything) without changing anything else —
every downstream consumer just sees the same pose dict format
regardless of whether it came from the procedural generator
(skeleton.generate_pose) or real AIST++ data.

This is tested against synthetic COCO-shaped data (known coordinates,
not real AIST++ data — that still needs to be downloaded separately;
see the project roadmap / GitHub issue) so the mapping logic itself is
verified before a real downloaded sequence is available to test with.
"""

import numpy as np

# Standard COCO 17-keypoint order — this is the exact order AIST++'s
# keypoints3d/keypoints2d files use.
COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def coco_frame_to_pose(coco_kpts: np.ndarray) -> dict:
    """
    Convert one frame of COCO-format keypoints into this project's pose
    dict format (see skeleton.SKELETON_JOINTS).

    Args:
        coco_kpts: array of shape (17, 2) or (17, 3) in COCO_JOINTS
            order — only the first two columns (x, y) are used; a third
            column (z depth, or confidence for 2D data) is ignored here.
            2D projection of 3D data is a separate, later concern.

    Returns:
        dict mapping our joint names -> [x, y]
    """
    def get(name):
        return coco_kpts[COCO_JOINTS.index(name)][:2]

    l_sh, r_sh = get("left_shoulder"), get("right_shoulder")
    l_hip, r_hip = get("left_hip"), get("right_hip")

    pose = {
        "head": get("nose"),
        "neck": (l_sh + r_sh) / 2.0,          # COCO has no neck joint — derived
        "l_shoulder": l_sh,
        "r_shoulder": r_sh,
        "l_elbow": get("left_elbow"),
        "r_elbow": get("right_elbow"),
        "l_hand": get("left_wrist"),
        "r_hand": get("right_wrist"),
        "hip_center": (l_hip + r_hip) / 2.0,  # COCO has no hip-center — derived
        "l_hip": l_hip,
        "r_hip": r_hip,
        "l_knee": get("left_knee"),
        "r_knee": get("right_knee"),
        "l_foot": get("left_ankle"),
        "r_foot": get("right_ankle"),
    }
    return {k: [float(v[0]), float(v[1])] for k, v in pose.items()}


def coco_sequence_to_poses(coco_seq: np.ndarray) -> list:
    """Convert a full (num_frames, 17, 2+) sequence into a list of pose dicts."""
    return [coco_frame_to_pose(frame) for frame in coco_seq]