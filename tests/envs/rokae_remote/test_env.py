import numpy as np
import torch

from rlinf.envs.rokae_remote.rokae_remote_env import RokaeRemoteEnv


class _FakeGateway:
    def __init__(self):
        self.step_count = 0
        self.commands = []
        self.stop_on_step = False

    @staticmethod
    def _observation():
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        return {
            "state": np.arange(7, dtype=np.float32),
            "images": {"external": image, "wrist": image + 1},
        }

    def request(self, command, **payload):
        del payload
        self.commands.append(command)
        if command == "reset":
            return {"observation": self._observation()}
        if command == "step":
            self.step_count += 1
            response = {"observation": self._observation(), "reward": 0.0}
            if self.stop_on_step:
                response.update(
                    {"terminated": True, "info": {"episode_command": "stop"}}
                )
            return response
        raise AssertionError(command)


def _make_env(execution_horizon=30):
    env = RokaeRemoteEnv.__new__(RokaeRemoteEnv)
    env.action_dim = 7
    env.main_image_key = "external"
    env.extra_image_keys = ["wrist"]
    env.task_description = "test"
    env.execution_horizon = execution_horizon
    env.auto_reset = True
    env.ignore_terminations = False
    env.max_episode_steps = None
    env.client = _FakeGateway()
    env._is_start = True
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env._return = np.zeros(1, dtype=np.float32)
    env._success_once = np.zeros(1, dtype=bool)
    env._intervened_once = np.zeros(1, dtype=bool)
    env._intervened_steps = np.zeros(1, dtype=np.int32)
    return env


def test_wrap_obs_exposes_shared_openpi_camera_keys():
    env = _make_env()
    obs = env._wrap_obs(env.client._observation())

    assert obs["main_images"].shape == (1, 4, 5, 3)
    assert obs["extra_view_images"].shape == (1, 1, 4, 5, 3)


def test_chunk_step_pads_transition_and_expert_tensors_to_policy_horizon():
    env = _make_env(execution_horizon=30)
    actions = torch.zeros((1, 50, 7), dtype=torch.float32)

    obs_list, rewards, terms, truncs, infos = env.chunk_step(actions)

    assert len(obs_list) == 30
    assert env.client.step_count == 30
    assert rewards.shape == (1, 50)
    assert terms.shape == (1, 50)
    assert truncs.shape == (1, 50)
    assert infos[-1]["intervene_action"].shape == (1, 350)
    assert infos[-1]["intervene_flag"].shape == (1, 50)
    assert not infos[-1]["intervene_flag"].any()


def test_stop_command_does_not_issue_reset_after_gateway_shutdown():
    env = _make_env(execution_horizon=1)
    env.client.stop_on_step = True

    _, _, terminations, _, info = env.step(torch.zeros(7), auto_reset=True)

    assert terminations.item()
    assert env.client.commands == ["step"]
    assert info["episode_command"] == "stop"


def test_chunk_terminal_is_aligned_to_last_executed_frame():
    env = _make_env(execution_horizon=30)
    env.client.stop_on_step = True

    obs_list, rewards, terms, truncs, infos = env.chunk_step(
        torch.zeros((1, 50, 7), dtype=torch.float32)
    )

    assert len(obs_list) == 1
    assert env.client.step_count == 1
    assert rewards.shape == (1, 50)
    assert terms[0, 0].item()
    assert not terms[0, 1:].any()
    assert not truncs.any()
    assert infos[-1]["episode_command"] == "stop"


def test_early_terminal_still_pads_to_full_policy_horizon():
    env = _make_env(execution_horizon=30)
    env.client.stop_on_step = True

    _, rewards, terms, _, infos = env.chunk_step(
        torch.zeros((1, 50, 7), dtype=torch.float32)
    )

    assert rewards.shape == (1, 50)
    assert terms.shape == (1, 50)
    assert infos[-1]["intervene_action"].shape == (1, 350)
    assert infos[-1]["intervene_flag"].shape == (1, 50)
