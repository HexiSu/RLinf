# Copyright 2026 The RLinf Authors.

"""Reliable request/reply client for the ROKAE hardware gateway."""

from __future__ import annotations

from typing import Any

import zmq

from .protocol import PROTOCOL_VERSION, pack_message, unpack_message


class RokaeGatewayError(RuntimeError):
    pass


class RokaeGatewayClient:
    def __init__(self, address: str, timeout_ms: int = 3000):
        self.address = address
        self.timeout_ms = int(timeout_ms)
        self._context = zmq.Context.instance()
        self._socket = None
        self._request_id = 0
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.connect(self.address)

    def request(self, command: str, **payload: Any) -> dict[str, Any]:
        self._request_id += 1
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self._request_id,
            "command": command,
            **payload,
        }
        try:
            self._socket.send(pack_message(request))
            response = unpack_message(self._socket.recv())
        except zmq.Again as exc:
            self._connect()
            raise RokaeGatewayError(
                f"ROKAE gateway timed out at {self.address} during {command!r}"
            ) from exc
        if response.get("request_id") != self._request_id:
            raise RokaeGatewayError("ROKAE gateway returned a mismatched request_id")
        if not response.get("ok", False):
            raise RokaeGatewayError(str(response.get("error", "gateway request failed")))
        return response

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self.request("close_client")
        except Exception:
            pass
        self._socket.close(linger=0)
        self._socket = None
