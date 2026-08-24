# Copyright 2026 The RLinf Authors.

import numpy as np

from rlinf.models.embodiment.openpi.policies.rokae_joint_policy import (
    RAW_SELECTION,
    select_joint_gripper,
)


def test_selects_joint_and_gripper_from_raw_rokae_vector():
    raw = np.arange(28, dtype=np.float32).reshape(2, 14)
    selected = select_joint_gripper(raw)
    np.testing.assert_array_equal(selected, raw[:, list(RAW_SELECTION)])


def test_keeps_canonical_online_action():
    canonical = np.arange(21, dtype=np.float32).reshape(3, 7)
    np.testing.assert_array_equal(select_joint_gripper(canonical), canonical)
