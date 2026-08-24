#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.

"""Convert robot-local .npz episodes into RLinf LeRobot shards."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from rlinf.data.storage.lerobot import LeRobotDatasetWriter


def convert(source: pathlib.Path, output: pathlib.Path, fps: int) -> pathlib.Path:
    with np.load(source, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        external = np.asarray(data["images_external"])
        wrist = np.asarray(data["images_wrist"])
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
        raise ValueError(f"invalid episode arrays in {source}")
    if len(states) == 0:
        raise ValueError(f"empty episode: {source}")
    shard = output / source.stem
    if shard.exists():
        return shard
    frames = []
    success = bool(metadata.get("success", False))
    for index in range(len(states)):
        frame = {
            "state": states[index],
            "actions": actions[index],
            "done": np.asarray([index == len(states) - 1]),
            "is_success": np.asarray([success and index == len(states) - 1]),
            "intervene_flag": np.asarray([False]),
            "task": str(metadata.get("task_prompt", "")),
        }
        if external.ndim >= 4 and external.shape[0] == len(states):
            frame["image"] = external[index]
        if wrist.ndim >= 4 and wrist.shape[0] == len(states):
            frame["wrist_image"] = wrist[index]
        frames.append(frame)
    writer = LeRobotDatasetWriter()
    first = frames[0]
    writer.create(
        repo_id=str(shard),
        robot_type="rokae_robot",
        fps=fps,
        image_shape=first["image"].shape if "image" in first else (1, 1, 3),
        state_dim=int(states.shape[-1]),
        action_dim=int(actions.shape[-1]),
        has_image="image" in first,
        wrist_image_keys={"wrist_image": first["wrist_image"].shape}
        if "wrist_image" in first
        else None,
        has_intervene_flag=True,
    )
    writer.add_episode(frames)
    writer.finalize()
    return shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-s", type=float, default=2.0)
    args = parser.parse_args()
    source_root = pathlib.Path(args.input).expanduser().resolve()
    output = pathlib.Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    while True:
        for source in sorted(source_root.glob("*.npz")):
            try:
                print(f"Converting {source}")
                print(f"Created {convert(source, output, args.fps)}")
                source.unlink()
            except Exception as exc:
                print(f"Conversion failed for {source}: {exc}")
        if not args.watch:
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
