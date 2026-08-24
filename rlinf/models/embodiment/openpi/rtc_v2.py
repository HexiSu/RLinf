# Copyright 2026 The RLinf Authors.

"""PyTorch port of the RTCv2 semantics used by the ROKAE JAX checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from openpi.models_pytorch.pi0_pytorch import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)


class RTCGuidanceContext:
    def __init__(
        self,
        prev_model_actions=None,
        executed_horizon: int = 0,
        delay_steps: int = 0,
    ):
        self.prev_model_actions = prev_model_actions
        self.executed_horizon = int(executed_horizon)
        self.delay_steps = int(delay_steps)


def prefix_attention_end(start, max_end, mode, multiplier, offset):
    start = torch.as_tensor(start)
    max_end = torch.as_tensor(max_end, device=start.device)
    if mode == "fixed":
        return max_end
    if mode == "scaled":
        scaled = torch.ceil(start.float() * float(multiplier)).to(torch.int64)
        return torch.where(
            start == 0,
            torch.zeros_like(start),
            torch.minimum(max_end, torch.maximum(start, scaled)),
        )
    if mode == "offset":
        shifted = start + int(offset)
        return torch.where(
            start == 0,
            torch.zeros_like(start),
            torch.minimum(max_end, torch.maximum(start, shifted)),
        )
    raise ValueError(f"Invalid RTC prefix attention end mode: {mode}")


def prefix_weights(start, end, total: int, schedule: str, *, dtype=torch.float32):
    start = torch.as_tensor(start)
    end = torch.as_tensor(end, device=start.device)
    start = torch.minimum(start, end)
    index = torch.arange(total, device=start.device)
    while index.ndim < start.ndim + 1:
        index = index.unsqueeze(0)
    start_e = start.unsqueeze(-1)
    end_e = end.unsqueeze(-1)
    if schedule == "ones":
        weights = torch.ones_like(index + start_e, dtype=dtype)
    elif schedule == "zeros":
        weights = (index < start_e).to(dtype)
    elif schedule in {"linear", "exp"}:
        weights = torch.clamp(
            (start_e - 1 - index) / (end_e - start_e + 1).to(torch.float32) + 1,
            0,
            1,
        ).to(dtype)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / (math.e - 1.0)
    else:
        raise ValueError(f"Invalid RTC prefix attention schedule: {schedule}")
    return torch.where(index >= end_e, torch.zeros_like(weights), weights)


def configured_prefix_weights(
    model, delay, total: int, overlap_horizon=None, dtype=torch.float32
):
    horizon = getattr(model.config, "rtc_prefix_attention_horizon", None)
    horizon = model.config.max_delay if horizon is None else int(horizon)
    max_end = min(total, horizon)
    if overlap_horizon is not None:
        max_end = min(max_end, int(overlap_horizon))
    delay = torch.as_tensor(delay, dtype=torch.int64)
    max_end_tensor = torch.full_like(delay, max_end)
    end = prefix_attention_end(
        delay,
        max_end_tensor,
        model.config.rtc_prefix_attention_end_mode,
        model.config.rtc_prefix_attention_delay_multiplier,
        model.config.rtc_prefix_attention_offset,
    )
    return prefix_weights(
        delay, end, total, model.config.rtc_prefix_attention_schedule, dtype=dtype
    )


def embed_suffix_rtc(model, state, noisy_actions, timestep):
    """PI05 suffix embedding with a separate flow timestep for each action token."""
    if not model.pi05 or timestep.ndim == 1:
        return model.embed_suffix(state, noisy_actions, timestep)
    batch, horizon = timestep.shape
    flat_time = timestep.reshape(-1)
    time_emb = create_sinusoidal_pos_embedding(
        flat_time,
        model.action_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0,
        device=timestep.device,
    ).reshape(batch, horizon, -1)
    time_emb = time_emb.to(dtype=timestep.dtype)
    action_emb = model._apply_checkpoint(model.action_in_proj, noisy_actions)

    def time_mlp(value):
        value = F.silu(model.time_mlp_in(value))
        return F.silu(model.time_mlp_out(value))

    adarms_cond = model._apply_checkpoint(time_mlp, time_emb)
    pad_masks = torch.ones(
        batch, horizon, dtype=torch.bool, device=noisy_actions.device
    )
    att_masks = torch.tensor(
        [1] + [0] * (horizon - 1),
        dtype=action_emb.dtype,
        device=action_emb.device,
    )[None].expand(batch, horizon)
    return action_emb, pad_masks, att_masks, adarms_cond


def suffix_velocity(model, state, prefix_pad_masks, past_key_values, x_t, timestep):
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = embed_suffix_rtc(
        model, state, x_t, timestep
    )
    suffix_len = suffix_pad_masks.shape[1]
    prefix_masks = prefix_pad_masks[:, None, :].expand(
        prefix_pad_masks.shape[0], suffix_len, prefix_pad_masks.shape[1]
    )
    suffix_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_mask = torch.cat([prefix_masks, suffix_masks], dim=2)
    position_ids = (
        prefix_pad_masks.sum(dim=-1)[:, None]
        + torch.cumsum(suffix_pad_masks, dim=1)
        - 1
    )
    full_mask = model._prepare_attention_masks_4d(full_mask)
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    outputs, _ = model.paligemma_with_expert.forward(
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs[1][:, -x_t.shape[1] :].to(torch.float32)
    return model.action_out_proj(suffix_out)


def rtc_sft_loss(model, observation, actions):
    """RTCv2 flow-matching loss, including random delay and token-wise weights."""
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=True
    )
    batch, horizon, action_dim = actions.shape
    noise = model.sample_noise(actions.shape, actions.device)
    time = model.sample_time(batch, actions.device)
    fixed_delay = getattr(model.config, "rtc_train_delay", None)
    if fixed_delay is None:
        delay = torch.randint(
            0, int(model.config.max_delay), (batch,), device=actions.device
        )
    else:
        delay = torch.full((batch,), int(fixed_delay), device=actions.device)
    weights = configured_prefix_weights(model, delay, horizon, dtype=actions.dtype)
    time_masked = (1 - weights) * time[:, None]
    x_t = time_masked[..., None] * noise + (1 - time_masked[..., None]) * actions
    target_velocity = noise - actions
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = embed_suffix_rtc(
        model, state, x_t, time_masked
    )
    backbone_dtype = model.paligemma_with_expert.paligemma.language_model.layers[
        0
    ].self_attn.q_proj.weight.dtype
    prefix_embs = prefix_embs.to(backbone_dtype)
    suffix_embs = suffix_embs.to(backbone_dtype)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    outputs, _ = model.paligemma_with_expert.forward(
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    velocity = model.action_out_proj(outputs[1][:, -horizon:].to(torch.float32))
    element_loss = F.mse_loss(velocity, target_velocity, reduction="none")
    loss_weights = (1 - weights)[..., None]
    # Scale so the caller's ordinary .mean() exactly matches the JAX weighted
    # sum over action dimensions divided by the sum of token weights.
    scale = (horizon * action_dim) / loss_weights.sum(dim=(1, 2)).clamp_min(1e-8)
    return element_loss * loss_weights * scale[:, None, None]


@torch.no_grad()
def sample_actions_with_rtc_guidance(
    model,
    observation,
    rtc_context,
    noise=None,
    mode="eval",
    compute_values=True,
):
    del mode, compute_values
    batch = observation.state.shape[0]
    device = observation.state.device
    horizon = model.config.action_horizon
    if noise is None:
        noise = model.sample_noise((batch, horizon, model.config.action_dim), device)
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False
    )
    _, prefix_pad_masks, cache = model._build_prefix_cache(
        images, img_masks, lang_tokens, lang_masks
    )
    rate = int(model.config.rtc_rate_of_inference)
    previous = rtc_context.prev_model_actions.to(device=device, dtype=noise.dtype)
    aligned = previous[:, rate:, :]
    aligned = F.pad(aligned, (0, 0, 0, max(horizon - aligned.shape[1], 0)))[:, :horizon]
    delay = torch.full(
        (batch,),
        min(max(int(rtc_context.delay_steps), 0), int(model.config.max_delay)),
        device=device,
        dtype=torch.int64,
    )
    weights = configured_prefix_weights(
        model,
        delay,
        horizon,
        overlap_horizon=max(horizon - rate, 0),
        dtype=noise.dtype,
    )
    x_t = noise
    chains = [x_t]
    num_steps = int(model.config.num_steps)
    for index in range(num_steps):
        x_t = weights[..., None] * aligned + (1 - weights[..., None]) * x_t
        scalar_time = 1.0 - index / num_steps
        token_time = (1 - weights) * scalar_time
        velocity = suffix_velocity(
            model, state, prefix_pad_masks, cache, x_t, token_time
        )
        x_t = x_t - velocity / num_steps
        chains.append(x_t)
    chunk = int(model.config.action_chunk)
    env_dim = int(model.config.action_env_dim)
    return {
        "actions": x_t,
        "chains": torch.stack(chains, dim=1),
        "prev_logprobs": torch.zeros((batch, chunk, env_dim), device=device),
        "prev_values": torch.zeros((batch, 1), device=device),
        "denoise_inds": torch.full(
            (batch, num_steps), -1, device=device, dtype=torch.long
        ),
    }
