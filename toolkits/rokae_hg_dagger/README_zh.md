# ROKAE Pi05 本地 Rollout HG-DAgger

本流程只支持本地 rollout：机器人电脑负责相机采集、Pi05 推理、人工接管和 30Hz
机械臂控制；训练服务器负责接收 episode、运行 RLinf 训练和发布新 checkpoint。

## 目录

```text
<RUN_ROOT>/
  jax_step3000/
  pi05_step3000_torch/
  robot_episodes/
  published/latest
  rokae_hg_dagger_pi05_step3000/checkpoints/
```

`/vepfs-1/users/piaoweiyi/Projects/openpi` 只读使用，不写入训练结果。

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

## 2. 启动训练服务器服务

终端一：

```bash
python toolkits/rokae_hg_dagger/trajectory_receiver.py \
  --root <RUN_ROOT>/robot_episodes --host 0.0.0.0 --port 8766
```

终端二：

```bash
python toolkits/rokae_hg_dagger/policy_sync_server.py \
  --latest <RUN_ROOT>/published/latest --host 0.0.0.0 --port 8765
```

只开放机器人电脑到训练服务器的 TCP `8765`、`8766`。

## 3. 发布初始策略

```bash
bash toolkits/rokae_hg_dagger/publish_policy_checkpoint.sh \
  <RUN_ROOT>/pi05_step3000_torch <RUN_ROOT>/published
```

该命令原子更新 `published/latest`。

## 4. 启动机器人本地 rollout

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

每个 episode 结束后，机器人保存并上传状态、动作、双路图像和任务文本。上传失败时
文件保留在 `trajectory_spool_dir`。

## 5. 启动 RLinf 训练

确认 `robot_episodes/` 已收到 episode 后，在训练服务器提交：

```bash
cd /vepfs-1/users/sunhaoxuan/RLinf
ROBOT_PC_IP=<ROBOT_IP> \
bash toolkits/rokae_hg_dagger/run_server_hg_dagger.sh train
```

续训：

```bash
RESUME_DIR=<RUN_ROOT>/rokae_hg_dagger_pi05_step3000/checkpoints/global_step_<N> \
ROBOT_PC_IP=<ROBOT_IP> \
bash toolkits/rokae_hg_dagger/run_server_hg_dagger.sh train
```

## 6. 发布训练后的策略

```bash
bash toolkits/rokae_hg_dagger/publish_policy_checkpoint.sh \
  <RUN_ROOT>/rokae_hg_dagger_pi05_step3000/checkpoints/global_step_<N>/actor \
  <RUN_ROOT>/published
```

机器人只在下一个 episode 开始时检查 `published/latest`，下载并校验成功后才切换；
episode 结束不会自动触发策略更新。

## 数据格式

`trajectory_receiver.py` 接收压缩 `.npz`，包含 `states`、`actions`、
`images_external`、`images_wrist` 和 `metadata`。接入 `online_lerobot` actor 前，
需要将 `.npz` 转成 LeRobot episode shard，或配置等价的数据加载器。

首次真机运行请使用 `--mode test` 验证动作维度和相机键名，再切换到 autonomous，并
降低机械臂速度/加速度，确保急停可用。
