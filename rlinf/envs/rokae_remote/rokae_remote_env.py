"""Single-robot RLinf environment backed by a remote ROKAE gateway."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .client import RokaeGatewayClient


def _tensor(value, dtype=None):
    return torch.as_tensor(value, dtype=dtype)


class RokaeRemoteEnv(gym.Env):
    """Expose one physical 6-DoF arm plus gripper as an RLinf vector env."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        del seed_offset, total_num_processes, worker_info
        if num_envs != 1:
            raise ValueError("rokae_remote supports exactly one environment per worker")
        self.cfg = cfg
        self.num_envs = 1
        self.auto_reset = bool(cfg.get("auto_reset", True))
        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self.max_episode_steps = cfg.get("max_episode_steps", None)
        self.main_image_key = str(cfg.get("main_image_key", "external"))
        self.extra_image_keys = list(cfg.get("extra_image_keys", ["wrist"]))
        self.task_description = str(cfg.task_description)
        self.execution_horizon = int(cfg.get("execution_horizon", 30))
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        gateway_cfg = cfg.get("gateway", {})
        self.client = RokaeGatewayClient(
            str(gateway_cfg.get("address", "tcp://127.0.0.1:5560")),
            int(gateway_cfg.get("timeout_ms", 3000)),
        )
        hello = self.client.request("hello")
        self.action_dim = int(hello["action_dim"])
        if self.action_dim != 7 or int(hello["joint_num"]) != 6:
            raise ValueError(
                "This config requires a 6-DoF arm plus gripper (7D action); "
                f"gateway reported joint_num={hello['joint_num']}, action_dim={self.action_dim}"
            )
        camera_shapes = hello.get("camera_shapes", {})
        missing = [
            key
            for key in [self.main_image_key, *self.extra_image_keys]
            if key not in camera_shapes
        ]
        if missing:
            raise KeyError(f"Gateway is missing required cameras: {missing}")
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, self.action_dim), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})
        self._is_start = True
        self._elapsed_steps = np.zeros(1, dtype=np.int32)
        self._return = np.zeros(1, dtype=np.float32)
        self._success_once = np.zeros(1, dtype=bool)
        self._intervened_once = np.zeros(1, dtype=bool)
        self._intervened_steps = np.zeros(1, dtype=np.int32)

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2

    def update_reset_state_ids(self):
        return None

    def _reset_metrics(self):
        self._elapsed_steps[:] = 0
        self._return[:] = 0
        self._success_once[:] = False
        self._intervened_once[:] = False
        self._intervened_steps[:] = 0

    def _wrap_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(raw["state"], dtype=np.float32).reshape(1, self.action_dim)
        images = raw["images"]
        obs = {
            "states": _tensor(state, torch.float32),
            "main_images": _tensor(np.asarray(images[self.main_image_key])[None]),
            "task_descriptions": [self.task_description],
        }
        if self.extra_image_keys:
            extra_images = [
                np.asarray(images[key]) for key in self.extra_image_keys
            ]
            extra = np.stack(extra_images, axis=0)
            obs["extra_view_images"] = _tensor(extra[None])
        else:
            obs["extra_view_images"] = None
        return obs

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        del reset_state_ids, seed, options, env_idx
        response = self.client.request("reset")
        self._reset_metrics()
        return self._wrap_obs(response["observation"]), {}

    def _episode_info(self, reward: float, intervened: bool) -> dict[str, Any]:
        self._return[0] += reward
        self._success_once[0] |= bool(np.isclose(reward, 1.0))
        self._intervened_once[0] |= intervened
        self._intervened_steps[0] += int(intervened)
        length = max(int(self._elapsed_steps[0]), 1)
        return {
            "success_once": _tensor(self._success_once.copy()),
            "success_at_end": _tensor(self._success_once.copy()),
            "return": _tensor(self._return.copy()),
            "episode_len": _tensor(self._elapsed_steps.copy()),
            "reward": _tensor(self._return.copy() / length),
            "intervened_once": _tensor(self._intervened_once.copy()),
            "intervened_steps": _tensor(self._intervened_steps.copy()),
            "success_no_intervened": _tensor(
                self._success_once.copy() & ~self._intervened_once.copy()
            ),
        }

    def step(self, actions, auto_reset=True):
        array = np.asarray(
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else actions,
            dtype=np.float32,
        ).reshape(1, self.action_dim)
        response = self.client.request("step", policy_action=array[0])
        self._elapsed_steps += 1
        reward = float(response.get("reward", 0.0))
        terminated = bool(response.get("terminated", False))
        truncated = bool(response.get("truncated", False))
        if (
            self.max_episode_steps is not None
            and self._elapsed_steps[0] >= int(self.max_episode_steps)
        ):
            truncated = True
        expert = response.get("intervene_action")
        intervened = expert is not None
        expert_action = np.zeros((1, self.action_dim), dtype=np.float32)
        if intervened:
            expert_action[0] = np.asarray(expert, dtype=np.float32).reshape(
                self.action_dim
            )
        info = {
            "intervene_action": _tensor(expert_action, torch.float32),
            "intervene_flag": _tensor([intervened], torch.bool),
        }
        info.update(response.get("info", {}))
        info["episode"] = self._episode_info(reward, intervened)
        if self.ignore_terminations:
            terminated = False
        obs = self._wrap_obs(response["observation"])
        terminations = _tensor([terminated], torch.bool)
        truncations = _tensor([truncated], torch.bool)
        rewards = _tensor([reward], torch.float32)
        stop_requested = info.get("episode_command") == "stop"
        if (
            (terminated or truncated)
            and auto_reset
            and self.auto_reset
            and not stop_requested
        ):
            obs, info = self._handle_auto_reset(obs, info)
        return obs, rewards, terminations, truncations, info

    def _handle_auto_reset(self, final_obs, final_info):
        saved_obs = copy.deepcopy(final_obs)
        saved_info = copy.deepcopy(final_info)
        obs, info = self.reset()
        info.update(
            {
                "final_observation": saved_obs,
                "final_info": saved_info,
                "_final_info": np.array([True]),
                "_final_observation": np.array([True]),
                "_elapsed_steps": np.array([True]),
            }
        )
        return obs, info

    def chunk_step(self, chunk_actions):
        actions = (
            chunk_actions.detach().cpu()
            if isinstance(chunk_actions, torch.Tensor)
            else np.asarray(chunk_actions)
        )
        if (
            actions.ndim != 3
            or actions.shape[0] != 1
            or actions.shape[2] != self.action_dim
        ):
            raise ValueError(
                "chunk_actions must have shape [1, chunk_size, 7], "
                f"got {actions.shape}"
            )
        # The policy predicts the full RTC horizon (50 for this checkpoint),
        # while the robot executes only the configured inference window (30).
        # Keep transition tensors at the policy width so trajectory builders can
        # combine them with the 50-step action chunk; unexecuted tail entries
        # are zero-reward/non-terminal and have no expert takeover flag.
        model_chunk_size = int(actions.shape[1])
        if model_chunk_size <= 0:
            raise ValueError("chunk_actions must contain at least one action")
        chunk_size = min(model_chunk_size, self.execution_horizon)
        obs_list, infos_list = [], []
        rewards, terminations, truncations = [], [], []
        experts, flags = [], []
        done = False
        for index in range(chunk_size):
            obs, reward, terminated, truncated, info = self.step(
                actions[:, index], auto_reset=False
            )
            done = bool(terminated.any() or truncated.any())
            obs_list.append(obs)
            infos_list.append(info)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            experts.append(info["intervene_action"])
            flags.append(info["intervene_flag"])
            # Do not manufacture repeated observations after a physical
            # terminal/truncated transition.  The transition tensors below are
            # padded to the policy horizon, while the trajectory stream keeps
            # only frames that were actually returned by the gateway.
            if done:
                break
        executed_steps = len(rewards)
        executed_rewards = torch.stack(rewards, dim=1)
        executed_terms = torch.stack(terminations, dim=1)
        executed_truncs = torch.stack(truncations, dim=1)
        executed_experts = torch.stack(experts, dim=1)
        executed_flags = torch.stack(flags, dim=1)
        if model_chunk_size > executed_steps:
            tail = model_chunk_size - executed_steps
            zero_reward = torch.zeros((1, tail), dtype=executed_rewards.dtype)
            zero_done = torch.zeros((1, tail), dtype=torch.bool)
            zero_expert = torch.zeros(
                (1, tail, self.action_dim), dtype=executed_experts.dtype
            )
            zero_flag = torch.zeros((1, tail), dtype=torch.bool)
            chunk_rewards = torch.cat((executed_rewards, zero_reward), dim=1)
            chunk_terms = torch.cat((executed_terms, zero_done), dim=1)
            chunk_truncs = torch.cat((executed_truncs, zero_done), dim=1)
            chunk_experts = torch.cat((executed_experts, zero_expert), dim=1)
            chunk_flags = torch.cat((executed_flags, zero_flag), dim=1)
        else:
            chunk_rewards = executed_rewards
            chunk_terms = executed_terms
            chunk_truncs = executed_truncs
            chunk_experts = executed_experts
            chunk_flags = executed_flags
        past_done = torch.logical_or(chunk_terms, chunk_truncs).any(dim=1)
        chunk_experts = chunk_experts.reshape(1, -1)
        stop_requested = any(
            isinstance(info, dict) and info.get("episode_command") == "stop"
            for info in infos_list
        )
        if past_done.any() and self.auto_reset and not stop_requested:
            done_mask = torch.logical_or(chunk_terms, chunk_truncs)[0]
            terminal_index = int(torch.where(done_mask)[0][0])
            obs_list[-1], reset_info = self._handle_auto_reset(
                obs_list[terminal_index], infos_list[terminal_index]
            )
            reset_info["intervene_action"] = chunk_experts
            reset_info["intervene_flag"] = chunk_flags
            infos_list[-1] = reset_info
        else:
            # Keep the per-frame expert action in the terminal info when no
            # reset occurs (for example, the gateway requested Esc/stop).
            infos_list[-1]["intervene_action"] = chunk_experts
            infos_list[-1]["intervene_flag"] = chunk_flags
        # The trajectory builder iterates over ``obs_list`` (the physically
        # executed prefix), so expose a chunk-level done at its final frame.
        # This keeps the usual chunk-level contract while preserving terminal
        # episodes in online LeRobot data when the gateway ends early.
        collapsed_terms = torch.zeros_like(chunk_terms)
        collapsed_truncs = torch.zeros_like(chunk_truncs)
        if chunk_terms.any(dim=1).any():
            collapsed_terms[:, executed_steps - 1] = chunk_terms.any(dim=1)
        if chunk_truncs.any(dim=1).any():
            collapsed_truncs[:, executed_steps - 1] = chunk_truncs.any(dim=1)
        return obs_list, chunk_rewards, collapsed_terms, collapsed_truncs, infos_list

    def close(self):
        self.client.close()
