"""OpenPI transforms for a 6-DoF ROKAE arm with a scalar gripper."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

ROKAE_ACTION_DIM = 7
RAW_ROKAE_DIM = 14
RAW_SELECTION = (0, 1, 2, 3, 4, 5, 13)


def select_joint_gripper(value):
    """Accept either raw 14D ROKAE data or canonical 7D joint+gripper data."""
    if value.shape[-1] == ROKAE_ACTION_DIM:
        return value
    if value.shape[-1] == RAW_ROKAE_DIM:
        return value[..., list(RAW_SELECTION)]
    raise ValueError(
        f"Expected a 7D joint+gripper or raw 14D ROKAE vector, got {value.shape}"
    )


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC/CHW image, got {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RokaeJointInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state_source = data.get("state", data.get("observation/state"))
        if state_source is None:
            raise KeyError("Expected state or observation/state")
        state = select_joint_gripper(state_source)
        if "images" in data:
            base_image = _parse_image(data["images"]["external"])
            wrist_image = _parse_image(data["images"]["wrist"])
        else:
            base_image = _parse_image(data["observation/image"])
            wrist_source = data.get("observation/wrist_image")
            if wrist_source is None:
                wrist_source = data.get("observation/extra_view_image")
            if wrist_source is None:
                raise KeyError(
                    "Expected observation/wrist_image or observation/extra_view_image"
                )
            wrist_image = _parse_image(wrist_source)
        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # The original PI05 training transform keeps its zero-filled
                # third image unmasked.
                "right_wrist_0_rgb": np.False_
                if self.model_type == _model.ModelType.PI0
                else np.True_,
            },
        }
        if "actions" in data:
            actions = select_joint_gripper(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class RokaeJointOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ROKAE_ACTION_DIM])}
