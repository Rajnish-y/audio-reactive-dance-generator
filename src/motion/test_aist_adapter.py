"""
Test aist_adapter.py against synthetic COCO-shaped data (known
coordinates), since we don't have a real downloaded AIST++ sequence
yet. This verifies the mapping logic itself is correct, independent of
actually having real motion-capture data.
"""

import numpy as np
from aist_adapter import COCO_JOINTS, coco_frame_to_pose


def make_synthetic_frame():
    """
    Build one fake COCO frame with distinct, easy-to-check coordinates:
    joint index i gets position (i, i*10), so any mapping mistake is
    immediately obvious from the numbers.
    """
    frame = np.zeros((17, 3))
    for i in range(17):
        frame[i] = [i, i * 10, 0]
    return frame


def test_direct_mappings():
    frame = make_synthetic_frame()
    pose = coco_frame_to_pose(frame)

    checks = {
        "head": "nose",
        "l_shoulder": "left_shoulder",
        "r_shoulder": "right_shoulder",
        "l_elbow": "left_elbow",
        "r_elbow": "right_elbow",
        "l_hand": "left_wrist",
        "r_hand": "right_wrist",
        "l_hip": "left_hip",
        "r_hip": "right_hip",
        "l_knee": "left_knee",
        "r_knee": "right_knee",
        "l_foot": "left_ankle",
        "r_foot": "right_ankle",
    }

    for our_joint, coco_joint in checks.items():
        expected_idx = COCO_JOINTS.index(coco_joint)
        expected = [float(expected_idx), float(expected_idx * 10)]
        actual = pose[our_joint]
        assert actual == expected, (
            f"{our_joint}: expected {expected} (from COCO '{coco_joint}'), got {actual}"
        )
    print("All direct joint mappings correct.")


def test_derived_joints():
    frame = make_synthetic_frame()
    pose = coco_frame_to_pose(frame)

    l_sh_idx = COCO_JOINTS.index("left_shoulder")
    r_sh_idx = COCO_JOINTS.index("right_shoulder")
    expected_neck = [(l_sh_idx + r_sh_idx) / 2.0, (l_sh_idx + r_sh_idx) / 2.0 * 10]
    assert pose["neck"] == expected_neck, f"neck: expected {expected_neck}, got {pose['neck']}"

    l_hip_idx = COCO_JOINTS.index("left_hip")
    r_hip_idx = COCO_JOINTS.index("right_hip")
    expected_hip_center = [(l_hip_idx + r_hip_idx) / 2.0, (l_hip_idx + r_hip_idx) / 2.0 * 10]
    assert pose["hip_center"] == expected_hip_center, (
        f"hip_center: expected {expected_hip_center}, got {pose['hip_center']}"
    )
    print("Derived joints (neck, hip_center) correct.")


def test_all_our_joints_present():
    from skeleton import SKELETON_JOINTS
    frame = make_synthetic_frame()
    pose = coco_frame_to_pose(frame)
    missing = [j for j in SKELETON_JOINTS if j not in pose]
    assert not missing, f"Missing joints in adapter output: {missing}"
    print("All 15 skeleton joints present in adapter output.")


if __name__ == "__main__":
    test_direct_mappings()
    test_derived_joints()
    test_all_our_joints_present()
    print("\nAll adapter tests passed.")