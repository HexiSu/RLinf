# ROKAE Pi05 本地 Rollout HG-DAgger

本流程只支持本地 rollout：机器人电脑负责相机采集、Pi05 推理、人工接管和 30Hz
机械臂控制；训练服务器负责接收 episode、运行 RLinf 训练和发布新 checkpoint。

两侧完全解耦：机器人停止后，服务器继续使用已转换的 LeRobot 数据训练；服务器停止
或断连后，机器人继续使用当前本地模型，不再请求新策略或上传 episode，直到健康检查
确认连接恢复。

## 目录

```text
<RUN_ROOT>/
  jax_step3000/
  pi05_step3000_torch/
  robot_episodes/
  published/latest
  rokae_hg_dagger_pi05_step3000/checkpoints/
```

## 1. 准备初始 checkpoint

训练服务器执行一次：

```bash
cd /vepfs-1/users/sunhaoxuan/RLinf
bash toolkits/rokae_hg_dagger/run_server_hg_dagger.sh prepare
```

确认 `<RUN_ROOT>/pi05_step3000_torch/model.safetensors` 和 assets 存在，然后完整复制
到机器人电脑：

```bash
scp -r <RUN_ROOT>/pi05_step3000_torch robot@<ROBOT_IP>:/opt/rokae/policies/
```

## 2. 提交服务器任务

自动启动 episode 接收器、策略服务、`.npz` 转换器和 RLinf 训练；训练结束或任务被终止时，后台服务会统一清理：

```bash
cd /vepfs-1/users/sunhaoxuan/RLinf
ROBOT_PC_IP=<ROBOT_IP> \
bash toolkits/rokae_hg_dagger/run_server_hg_dagger.sh train
```

服务日志位于 `<RUN_ROOT>/trajectory_receiver.log`、`policy_sync_server.log` 和
`episode_converter.log`。只开放机器人电脑到训练服务器的 TCP `8765`、`8766`。

`prepare` 会同时生成模型种子：

```text
<RUN_ROOT>/rokae_hg_dagger_pi05_step3000/checkpoints/global_step_0/actor_seed.pt
```

该文件是 step-3000 Pi05 权重的 model-only seed，不包含优化器状态。训练入口会用
`runner.ckpt_path` 加载它，global step 从 0 开始；之后保存的正常 checkpoint 才使用
标准 `RESUME_DIR`。

## 3. 启动机器人本地 rollout

机器人电脑先启动本机 `rokae_zmq_server`、相机和机械臂驱动，再执行：

```bash
cd /path/to/lerobot_rokae
source /path/to/robot-venv/bin/activate
export PYTHONPATH=/path/to/RLinf:${PYTHONPATH:-}

python -m rokae_policy_runtime.openpi.cli \
  --bridge async_bridge \
  --local_checkpoint /opt/rokae/policies/pi05_step3000_torch \
  --local_device cuda:0 \
  --checkpoint_sync_endpoint http://<TRAIN_SERVER_IP>:8765 \
  --checkpoint_sync_root /opt/rokae/policies/versions \
  --trajectory_upload_endpoint http://<TRAIN_SERVER_IP>:8766 \
  --trajectory_spool_dir /opt/rokae/policies/episode_spool \
  --cam_high_serial <EXTERNAL_SERIAL> \
  --cam_wrist_serial <WRIST_SERIAL> \
  --control_freq 30
```

机器人命令中的 `--local_checkpoint` 只是启动兜底模型。只要配置了
`--checkpoint_sync_endpoint`，后续 episode 会按服务器 `published/latest` 的版本切换，
不会每次续训都重新固定加载 step-3000。每个 episode 结束后，机器人保存并上传状态、动作、双路图像和任务文本。上传失败时
文件保留在 `trajectory_spool_dir`。

## 4. 续训

```bash
RESUME_DIR=<RUN_ROOT>/rokae_hg_dagger_pi05_step3000/checkpoints/global_step_0 \
ROBOT_PC_IP=<ROBOT_IP> \
bash toolkits/rokae_hg_dagger/run_server_hg_dagger.sh train
```

首次训练也使用同一条命令；`global_step_0` 会自动识别为 model-only seed。训练中断后，
将 `RESUME_DIR` 改为最近的 `global_step_<N>` 即可恢复优化器和 global step。

## 5. 发布训练后的策略

单入口训练任务会自动运行 `watch_publish_policy.sh`。配置中的
`runner.save_interval=100` 保持不变；每个完整的 `global_step_N/actor` checkpoint
稳定写入后，发布器会自动更新 `published/latest`，无需人工执行发布命令。

发布器会先将 actor 的 `full_weights.pt` 导出成只包含推理所需的
`model.safetensors`、`config.json` 和 `assets/` 的目录，再由机器人同步；优化器状态、
FSDP shard 和训练元数据不会传到机器人。当前同步频率仍是每 100 个训练 step。

```bash
bash toolkits/rokae_hg_dagger/publish_policy_checkpoint.sh \
  <RUN_ROOT>/rokae_hg_dagger_pi05_step3000/checkpoints/global_step_<N>/actor \
  <RUN_ROOT>/published
```

机器人只在下一个 episode 开始时检查 `published/latest`，下载并校验成功后才切换；
episode 结束不会自动触发策略更新。

## 自动数据转换

服务器入口会自动运行 `convert_npz_episodes.py --watch`：每收到一个 `.npz`，立即
转换为 `<RUN_ROOT>/online_lerobot/<episode-id>/` LeRobot shard，再删除已成功转换的
传输文件。转换失败的文件会保留，错误写入 `episode_converter.log`，不会被训练读取。

首次真机运行请使用 `--mode test` 验证动作维度和相机键名，再切换到 autonomous，并
降低机械臂速度/加速度，确保急停可用。
