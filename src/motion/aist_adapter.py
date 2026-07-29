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


def resample_sequence(coco_seq: np.ndarray, native_fps: float, target_fps: float) -> np.ndarray:
    """
    Resample a (num_frames, 17, 3) sequence from its native frame rate
    (AIST++ is 60fps) to our pipeline's target frame rate (30fps by
    default, from config), via linear interpolation per joint/axis over
    time. This keeps real-time duration correct — a 12-second clip stays
    12 seconds, just represented with fewer (or more) frames — rather
    than naively dropping every other frame, which only works cleanly
    when the ratio is exactly 2:1 and gets wrong/jerky for anything else.
    """
    n_frames = coco_seq.shape[0]
    duration = n_frames / native_fps
    n_target = max(1, int(round(duration * target_fps)))

    src_times = np.arange(n_frames) / native_fps
    tgt_times = np.arange(n_target) / target_fps

    resampled = np.zeros((n_target, coco_seq.shape[1], coco_seq.shape[2]))
    for j in range(coco_seq.shape[1]):
        for k in range(coco_seq.shape[2]):
            resampled[:, j, k] = np.interp(tgt_times, src_times, coco_seq[:, j, k])
    return resampled


def normalize_pose_sequence(poses: list, target_height: float = 1.6, vertical_offset: float = 0.9) -> list:
    """
    Auto-scale and recenter a sequence of pose dicts from AIST++'s
    real-world units (roughly centimeters) to our renderer's expected
    coordinate range, based on the sequence's own bounding box — so any
    real clip fits the same visual frame our procedural motions do,
    without hardcoding a fixed scale that would only work for one clip.
    """
    all_xy = np.array([[v for v in pose.values()] for pose in poses])
    y_min, y_max = all_xy[:, :, 1].min(), all_xy[:, :, 1].max()
    center = all_xy.reshape(-1, 2).mean(axis=0)
    scale = target_height / (y_max - y_min) if y_max > y_min else 1.0

    def norm_pose(pose):
        return {
            j: [(v[0] - center[0]) * scale, (v[1] - center[1]) * scale + vertical_offset]
            for j, v in pose.items()
        }

    return [norm_pose(p) for p in poses]