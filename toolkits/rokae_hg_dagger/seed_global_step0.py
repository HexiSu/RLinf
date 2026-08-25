#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.

"""Create a model-only global_step_0 seed for RLinf continuation runs."""

from __future__ import annotations

import argparse
import pathlib

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, help="converted model.safetensors directory"
    )
    parser.add_argument("--output", required=True, help="global_step_0 directory")
    args = parser.parse_args()
    source = pathlib.Path(args.source).expanduser().resolve()
    output = pathlib.Path(args.output).expanduser().resolve()
    weights = source / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    target = output / "actor_seed.pt"
    output.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        state_dict = load_file(str(weights), device="cpu")
        torch.save(state_dict, target)
    (output / "README.txt").write_text(
        "Model-only seed. Use runner.ckpt_path for global_step_0; subsequent checkpoints use normal RESUME_DIR.\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
