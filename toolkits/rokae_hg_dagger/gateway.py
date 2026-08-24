#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.

"""ROKAE + SpaceMouse gateway for RLinf HG-DAgger.

Run this file in the ``lerobot_rokae`` environment on the robot computer.  It
imports that project but does not modify it.  The policy side connects through
ZeroMQ and never imports robot drivers.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import zmq

# Import registrations before draccus parses polymorphic device configs.
from lerobot.cameras.realsense.configuration_realsense import (  # noqa: F401
    RealSenseCameraConfig,
)
from lerobot.configs import parser
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot_robot_rokae.devices.rokae_robot.config_rokae_robot import (  # noqa: F401
    RokaeRobotConfig,
)
from lerobot_robot_rokae.record.rokae_record_plugin import (
    maybe_make_rokae_pipelines,
    reset_robot_and_grippers,
)
from lerobot_teleoperator_rokae.devices.spacemouse.config_spacemouse import (  # noqa: F401
    SpacemouseConfig,
)

from rlinf.envs.rokae_remote.protocol import (
    PROTOCOL_VERSION,
    pack_message,
    unpack_message,
)

LOG = logging.getLogger("rokae_hg_dagger_gateway")


@dataclass
class DatasetRuntimeConfig:
    fps: int = 30


@dataclass
class GatewayConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: DatasetRuntimeConfig
    bind_address: str = "tcp://0.0.0.0:5560"
    intervention_deadband: float = 1e-4
    intervention_latch_s: float = 0.5
    request_timeout_ms: int = 1000


class EpisodeKeyboard:
    """Thread-safe keyboard state: S success, F failure, R reset, Esc stop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: str | None = None
        self.listener = None

    def start(self):
        try:
            from pynput import keyboard
        except ImportError:
            LOG.warning("pynput is unavailable; use the episode_control RPC")
            return

        def on_press(key):
            try:
                char = key.char.lower()
            except AttributeError:
                char = "escape" if key == keyboard.Key.esc else ""
            mapping = {"s": "success", "f": "failure", "r": "reset", "escape": "stop"}
            if char in mapping:
                with self._lock:
                    self._pending = mapping[char]
                LOG.info("Keyboard episode command: %s", mapping[char])

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def pop(self) -> str | None:
        with self._lock:
            value, self._pending = self._pending, None
        return value

    def inject(self, command: str):
        if command not in {"success", "failure", "reset", "stop"}:
            raise ValueError(f"Unsupported episode command: {command}")
        with self._lock:
            self._pending = command

    def close(self):
        if self.listener is not None:
            self.listener.stop()


class RokaeGateway:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        if hasattr(cfg.robot, "control_loop_fps"):
            cfg.robot.control_loop_fps = cfg.dataset.fps
        self.robot = make_robot_from_config(cfg.robot)
        self.teleop = make_teleoperator_from_config(cfg.teleop)
        defaults = make_default_processors()
        pipelines = maybe_make_rokae_pipelines(cfg, self.robot, self.teleop, *defaults)
        if pipelines is None:
            raise ValueError(
                "ROKAE gateway requires the rokae_robot processor pipeline"
            )
        (
            self.teleop_action_processor,
            self.robot_action_processor,
            self.robot_observation_processor,
        ) = pipelines
        self.keyboard = EpisodeKeyboard()
        self.last_intervention_time = -np.inf
        self.last_request_id = 0
        self.running = True

    @staticmethod
    def _state(obs: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [*[obs[f"joint_pos{i}"] for i in range(6)], obs["gripper_pos"]],
            dtype=np.float32,
        )

    def _observation(self, raw_obs=None) -> dict[str, Any]:
        raw_obs = self.robot.get_observation() if raw_obs is None else raw_obs
        processed = self.robot_observation_processor(raw_obs)
        images = {key: np.asarray(processed[key]) for key in self.robot.cameras}
        return {"state": self._state(processed), "images": images}

    @staticmethod
    def _action_dict(action: np.ndarray) -> dict[str, float]:
        action = np.asarray(action, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(action)):
            raise ValueError("Policy action contains NaN or Inf")
        if not 0.0 <= action[6] <= 1.0:
            raise ValueError(f"gripper action must be in [0, 1], got {action[6]}")
        return {
            **{f"joint_pos{i}": float(action[i]) for i in range(6)},
            "gripper_pos": float(action[6]),
        }

    def _expert_action(self, obs: dict[str, Any]) -> tuple[np.ndarray, bool]:
        raw = self.teleop.get_action()
        velocity = np.asarray([raw[f"cart_vel{i}"] for i in range(6)], dtype=float)
        buttons = np.asarray(raw.get("buttons", []), dtype=float).reshape(-1)
        active_now = bool(
            np.max(np.abs(velocity)) > self.cfg.intervention_deadband
            or np.any(np.abs(buttons) > 0)
        )
        now = time.monotonic()
        if active_now:
            self.last_intervention_time = now
        active = (
            active_now
            or now - self.last_intervention_time <= self.cfg.intervention_latch_s
        )
        teleop_action = self.teleop_action_processor((raw, obs))
        final_action = self.robot_action_processor((teleop_action, obs))
        expert = np.asarray(
            [
                *[final_action[f"joint_pos{i}"] for i in range(6)],
                final_action["gripper_pos"],
            ],
            dtype=np.float32,
        )
        return expert, active

    def connect(self):
        if int(getattr(self.robot, "joint_num", -1)) != 6:
            raise ValueError(f"Expected a 6-DoF ROKAE arm, got {self.robot.joint_num}")
        self.robot.connect()
        self.teleop.connect()
        self.keyboard.start()

    def reset(self) -> dict[str, Any]:
        reset_robot_and_grippers(self.robot, self.teleop, self.teleop_action_processor)
        self.last_intervention_time = -np.inf
        return self._observation()

    def step(self, policy_action: np.ndarray) -> dict[str, Any]:
        obs_before = self.robot.get_observation()
        expert, intervened = self._expert_action(obs_before)
        selected = expert if intervened else np.asarray(policy_action, dtype=np.float32)
        selected_dict = self._action_dict(selected)
        final_action = self.robot_action_processor((selected_dict, obs_before))
        self.robot.send_action(final_action)
        command = self.keyboard.pop()
        reward = 1.0 if command == "success" else 0.0
        terminated = command in {"success", "failure", "reset", "stop"}
        if command == "stop":
            self.running = False
        return {
            "observation": self._observation(),
            "reward": reward,
            "terminated": terminated,
            "truncated": False,
            "intervene_action": expert if intervened else None,
            "info": {"episode_command": command or "continue"},
        }

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                f"Protocol mismatch: gateway={PROTOCOL_VERSION}, client={request.get('protocol_version')}"
            )
        request_id = int(request["request_id"])
        if request_id <= self.last_request_id:
            raise ValueError(f"Stale request_id {request_id} <= {self.last_request_id}")
        self.last_request_id = request_id
        command = request["command"]
        if command == "hello":
            return {
                "protocol_version": PROTOCOL_VERSION,
                "joint_num": 6,
                "action_dim": 7,
                "fps": self.cfg.dataset.fps,
                "camera_shapes": {
                    key: list(value) for key, value in self.robot._cameras_ft.items()
                },
            }
        if command == "reset":
            return {"observation": self.reset()}
        if command == "step":
            return self.step(request["policy_action"])
        if command == "episode_control":
            self.keyboard.inject(str(request["episode_command"]))
            return {}
        if command in {"heartbeat", "close_client"}:
            return {}
        raise ValueError(f"Unsupported gateway command: {command}")

    def serve(self):
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.cfg.request_timeout_ms)
        socket.bind(self.cfg.bind_address)
        LOG.info("ROKAE HG-DAgger gateway listening on %s", self.cfg.bind_address)
        while self.running:
            try:
                request = unpack_message(socket.recv())
            except zmq.Again:
                continue
            request_id = request.get("request_id")
            try:
                payload = self.dispatch(request)
                response = {"ok": True, "request_id": request_id, **payload}
            except Exception as exc:
                LOG.exception("Gateway request failed")
                response = {"ok": False, "request_id": request_id, "error": str(exc)}
            socket.send(pack_message(response))
        socket.close(linger=0)

    def close(self):
        self.keyboard.close()
        try:
            self.teleop.disconnect()
        finally:
            self.robot.disconnect()


@parser.wrap()
def main(cfg: GatewayConfig):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    gateway = RokaeGateway(cfg)

    def stop(*_):
        gateway.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    gateway.connect()
    try:
        gateway.serve()
    finally:
        gateway.close()


if __name__ == "__main__":
    main()
