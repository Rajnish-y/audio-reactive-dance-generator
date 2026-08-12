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
        coco_kpts: array of shape (17, 3) in COCO_JOINTS order — x, y,
            z (AIST++'s real depth axis). Previously only x, y were
            kept and z was dropped; now preserved for real 3D rendering
            instead of a flat plane.

    Returns:
        dict mapping our joint names -> [x, y, z]
    """
    def get(name):
        return coco_kpts[COCO_JOINTS.index(name)][:3]

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
    return {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in pose.items()}


def coco_sequence_to_poses(coco_seq: np.ndarray) -> list:
    """Convert a full (num_frames, 17, 2+) sequence into a list of pose dicts."""
    return [coco_frame_to_pose(frame) for frame in coco_seq]


def resample_sequence(coco_seq: np.ndarray, native_fps: float, target_fps: float, n_target: int = None) -> np.ndarray:
    """
    Resample a (num_frames, 17, 3) sequence from its native frame rate
    (AIST++ is 60fps) to our pipeline's target frame rate (30fps by
    default, from config), via linear interpolation per joint/axis over
    time. This keeps real-time duration correct — a 12-second clip stays
    12 seconds, just represented with fewer (or more) frames — rather
    than naively dropping every other frame, which only works cleanly
    when the ratio is exactly 2:1 and gets wrong/jerky for anything else.

    Args:
        n_target: if given, use this exact output frame count instead of
            deriving it from duration * target_fps. Needed by callers
            that are assembling many consecutive snippets (like beat
            intervals) and need to control cumulative rounding
            themselves — independently rounding each snippet's frame
            count causes small errors that accumulate across many
            snippets into a noticeable total drift.
    """
    n_frames = coco_seq.shape[0]
    duration = n_frames / native_fps
    if n_target is None:
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

    Scaling is uniform across x, y, AND z (all three multiplied by the
    same factor, derived from the y/height range) — using a different
    scale per axis would distort proportions, squashing or stretching
    depth relative to height rather than preserving the real body shape.
    """
    all_xyz = np.array([[v for v in pose.values()] for pose in poses])
    y_min, y_max = all_xyz[:, :, 1].min(), all_xyz[:, :, 1].max()
    center = all_xyz.reshape(-1, 3).mean(axis=0)
    scale = target_height / (y_max - y_min) if y_max > y_min else 1.0

    def norm_pose(pose):
        return {
            j: [
                (v[0] - center[0]) * scale,
                (v[1] - center[1]) * scale + vertical_offset,
                (v[2] - center[2]) * scale,
            ]
            for j, v in pose.items()
        }

    return [norm_pose(p) for p in poses]


def lerp_pose(pose_a: dict, pose_b: dict, alpha: float) -> dict:
    """Linearly interpolate between two poses, joint by joint. alpha=0 -> pose_a, alpha=1 -> pose_b."""
    return {
        j: [
            pose_a[j][0] * (1 - alpha) + pose_b[j][0] * alpha,
            pose_a[j][1] * (1 - alpha) + pose_b[j][1] * alpha,
            pose_a[j][2] * (1 - alpha) + pose_b[j][2] * alpha,
        ]
        for j in pose_a
    }


def ease_into_transition(poses_a: list, poses_b: list, blend_frames: int) -> list:
    """
    Concatenate two pose sequences with a smoothed entry into poses_b,
    instead of an instant cut — needed when switching between two
    DIFFERENT real clips, which (unlike our procedural sine motions)
    don't share a common neutral pose to cut cleanly at.

    Unlike a naive crossfade that trims frames from both sides (which
    silently shortens total duration — a real bug caught during testing:
    it broke the exact beat-alignment duration work from the previous
    step), this does NOT remove any frames from either sequence. It only
    blends the VALUES of poses_b's first `blend_frames` frames — from
    poses_a's last pose, easing toward poses_b's own trajectory — so
    total frame count (and therefore total duration) is completely
    unaffected by how many times a switch happens.
    """
    blend_frames = min(blend_frames, len(poses_b))
    if blend_frames <= 0 or not poses_a:
        return poses_a + poses_b

    last_pose_a = poses_a[-1]
    eased_b = list(poses_b)  # shallow copy — don't mutate the caller's list
    for i in range(blend_frames):
        alpha = (i + 1) / (blend_frames + 1)
        eased_b[i] = lerp_pose(last_pose_a, poses_b[i], alpha)

    return poses_a + eased_b


def compute_pose_gap(pose_a: dict, pose_b: dict) -> float:
    """Total joint displacement between two poses (sum of per-joint 3D distances)."""
    return sum(
        ((pose_b[j][0] - pose_a[j][0]) ** 2
         + (pose_b[j][1] - pose_a[j][1]) ** 2
         + (pose_b[j][2] - pose_a[j][2]) ** 2) ** 0.5
        for j in pose_a
    )


def adaptive_ease_into_transition(
    poses_a: list, poses_b: list,
    reference_frames: int = 10, min_blend_frames: int = 3, max_blend_frames: int = 30,
) -> list:
    """
    Like ease_into_transition, but chooses the blend length based on how
    large the actual pose gap is at THIS specific switch, instead of a
    fixed duration. Testing revealed a fixed short blend can still look
    like a fast snap when the gap happens to be unusually large — some
    switches measured 5-53x that clip's own normal per-frame movement,
    because each clip's playhead can land on an arbitrary, unrelated
    pose when a switch happens.

    The target per-frame speed during the blend is estimated from
    poses_a's own recent steady-state motion (average displacement over
    its last `reference_frames`), so the eased-in motion moves at a
    speed consistent with how that specific clip normally moves, rather
    than an arbitrary global constant. min/max bounds avoid a blend
    that's absurdly short (still a snap) or absurdly long (looks like
    slow-motion floating) for pathological gap sizes.

    This BOUNDS how bad a large gap can look — it does not eliminate the
    root cause. A real fix would search the target clip for the frame
    whose pose is closest to the current ending pose (nearest-pose /
    motion-graph matching) instead of always resuming at the playhead's
    arbitrary position. That's tracked as a separate future improvement,
    not attempted here — see the project roadmap / GitHub issue.
    """
    if not poses_a:
        return poses_b

    gap = compute_pose_gap(poses_a[-1], poses_b[0])

    ref = poses_a[-reference_frames - 1:] if len(poses_a) > reference_frames else poses_a
    if len(ref) >= 2:
        ref_speed = float(np.mean([compute_pose_gap(ref[i], ref[i + 1]) for i in range(len(ref) - 1)]))
        ref_speed = max(ref_speed, 1e-6)
        needed_frames = int(np.ceil(gap / ref_speed))
    else:
        # Not enough history to estimate a reliable reference speed (e.g.
        # the previous segment was only 1-2 frames long). Falling back to
        # ref_speed = gap would collapse needed_frames to the MINIMUM —
        # the opposite of safe. Use the maximum instead: without a
        # reliable estimate, a longer/slower blend is the conservative
        # choice, not a faster one.
        needed_frames = max_blend_frames

    blend_frames = int(np.clip(needed_frames, min_blend_frames, max_blend_frames))

    return ease_into_transition(poses_a, poses_b, blend_frames)