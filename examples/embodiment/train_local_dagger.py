# Copyright 2026 The RLinf Authors.

"""Train DAgger actor from locally archived LeRobot episodes only.

No environment or rollout worker is created. Robot-local rollout can stop or
disconnect while this process continues consuming already converted shards.
"""

from __future__ import annotations

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="rokae_hg_dagger_pi05")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    seed_resume = cfg.runner.get("resume_dir", None)
    if seed_resume and str(seed_resume).endswith("global_step_0"):
        seed_path = f"{seed_resume}/actor_seed.pt"
        import os

        if os.path.isfile(seed_path):
            cfg.runner.ckpt_path = seed_path
            cfg.runner.resume_dir = None
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
    from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
        AsyncEmbodiedDAGGERFSDPPolicy,
    )

    actor = AsyncEmbodiedDAGGERFSDPPolicy.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=placement
    )
    runner = OfflineRunner(cfg=cfg, actor=actor, env=None, rollout=None)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
