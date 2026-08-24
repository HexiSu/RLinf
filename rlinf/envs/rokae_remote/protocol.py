"""Wire protocol shared by the RLinf environment and the ROKAE gateway."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
_NDARRAY_MARKER = "__ndarray__"


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _NDARRAY_MARKER: True,
            "dtype": array.dtype.str,
            "shape": array.shape,
            "data": array.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get(_NDARRAY_MARKER) is True:
        array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"])).copy()
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def pack_message(message: dict[str, Any]) -> bytes:
    return msgpack.packb(_encode(message), use_bin_type=True)


def unpack_message(payload: bytes) -> dict[str, Any]:
    message = _decode(msgpack.unpackb(payload, raw=False))
    if not isinstance(message, dict):
        raise TypeError(f"ROKAE message must be a dict, got {type(message)!r}")
    return message
