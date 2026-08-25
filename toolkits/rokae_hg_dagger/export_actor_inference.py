#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.

"""Export an RLinf actor checkpoint as an OpenPI inference directory."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    parser.add_argument(
        "--base", required=True, help="initial pi05_step3000_torch directory"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    actor = pathlib.Path(args.actor).resolve()
    base = pathlib.Path(args.base).resolve()
    output = pathlib.Path(args.output).resolve()
    source = actor / "model_state_dict" / "full_weights.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    state = torch.load(source, map_location="cpu", weights_only=False)
    tensors = {
        key: value.contiguous()
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    save_file(tensors, str(output / "model.safetensors"))
    shutil.copy2(base / "config.json", output / "config.json")
    if (base / "assets").is_dir():
        shutil.copytree(base / "assets", output / "assets")
    (output / "inference_manifest.json").write_text(
        json.dumps(
            {"source_actor": str(actor), "tensor_count": len(tensors)}, indent=2
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
