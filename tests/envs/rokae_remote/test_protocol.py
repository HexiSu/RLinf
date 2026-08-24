# Copyright 2026 The RLinf Authors.

import numpy as np

from rlinf.envs.rokae_remote.protocol import pack_message, unpack_message


def test_protocol_roundtrip_numpy():
    message = {
        "request_id": 3,
        "action": np.arange(7, dtype=np.float32),
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "nested": {"flag": np.bool_(True)},
    }
    decoded = unpack_message(pack_message(message))
    assert decoded["request_id"] == 3
    np.testing.assert_array_equal(decoded["action"], message["action"])
    np.testing.assert_array_equal(decoded["image"], message["image"])
    assert decoded["nested"]["flag"] is True
