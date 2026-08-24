# ROKAE 单臂 π₀.₅ HG-DAgger

这套接入保持 `lerobot_rokae` 不变。机器人电脑运行 gateway；RLinf 所在训练机只通过
ZeroMQ TCP 发送 7 维绝对关节位置动作并接收相机、状态和人工干预标签。

## 已固定的数据与模型契约

- 机器人：单臂 6 DoF + 夹爪，`joint_position`。
- 策略状态/动作：`[joint_pos0..5, gripper_pos]`，共 7 维。
- 原始 14 维数据会选取索引 `[0,1,2,3,4,5,13]`；笛卡尔位姿和 `psi` 不进入策略。
- 相机：`external` 为主视角，`wrist` 为腕部视角，RGB `480x640`。
- 任务文本：`Install an orange cylindrical sleeve workpiece into the front-left slot of the base.`
- checkpoint：只使用 step `3000`。
- π₀.₅：action horizon `50`，模型内部 action dim `32`，环境输出 dim `7`；`discrete_state_input=false`，`max_token_len=180`。
- RTCv2：每 30 个控制步发起下一次推理，训练 `max_delay=6`（随机 0--5），prefix attention horizon=5、scaled/linear；部署 metadata 的最大 delay 为 5。

RLinf 环境每次向策略请求完整的 50 步 action chunk，但网关只执行前 30 个控制步；
后 20 步在 reward、done 和人工接管标记中补零，以保持 actor/trajectory builder 的 50 步
张量契约。在线 LeRobot episode 只记录真实执行的 30 帧，不会把补零尾部写入数据集。

## 1. 只读复制 checkpoint 3000

本流程只读使用训练 checkpoint 和 OpenPI 源码，不会写入
`/vepfs-1/users/piaoweiyi/Projects/openpi`。默认所有生成物统一放在：

```text
/vepfs-1/runs/schaeffler3d/
  schaeffler3d_cr_orange_round_front_left_pi05_rtc_hgdagger_v1_20260824_001/
```

其中包含 `jax_step3000/`、`pi05_step3000_torch/`、`online_lerobot/`、日志和
RLinf 训练 checkpoint。需要更换实验目录时设置 `RUN_NAME` 或 `RUN_ROOT`。

在训练机执行（`scp` 只读取服务器，不改远端）：

```bash
mkdir -p /LOCAL/checkpoints/pi05_rokae_jax
scp -P 11557 -r \
  root@115.191.63.139:/vepfs-1/runs/schaeffler3d/schaeffler3d_cr_orange_round_front_left_pi05_rtc_v1_20260811_001/checkpoints/schaeffler3d_cr_orange_round_front_left_pi05_rtc_v1_20260811_001/schaeffler3d_cr_orange_round_front_left_pi05_rtc_v1_20260811_001/3000 \
  /LOCAL/checkpoints/pi05_rokae_jax/
```

在 RLinf 的 OpenPI 环境中转换：

```bash
cd /home/suhexi/rokae/RLinf
python -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
  --checkpoint-dir /LOCAL/checkpoints/pi05_rokae_jax/3000 \
  --config-name pi05_rokae_joint_horizon50 \
  --output-path /LOCAL/checkpoints/pi05_rokae_step3000_torch \
  --precision bfloat16
```

转换器的 `--checkpoint-dir` 必须是包含 `params/` 和 `assets/` 的 checkpoint
根目录，不能直接指向 `params/`。

转换后必须检查：

```bash
cat /LOCAL/checkpoints/pi05_rokae_step3000_torch/config.json
test -f /LOCAL/checkpoints/pi05_rokae_step3000_torch/model.safetensors
test -f /LOCAL/checkpoints/pi05_rokae_step3000_torch/assets/schaeffler3d_cr_orange_round_front_left_robot1_cr_spacemouse_v1_20260811/2.1/norm_stats.json
```

`config.json` 应显示 `action_dim: 32`、`action_horizon: 50`、`discrete_state_input: false` 和 `max_token_len: 180`。不要用 5999 的目录。

## 2. 在机器人电脑启动 gateway

先按你现有流程启动 `rokae_zmq_server`。复制并编辑
`toolkits/rokae_hg_dagger/gateway_example.yaml`，至少替换两路相机序列号、底层机器人
ZMQ 端口和 reset trajectory。gateway 必须运行在已安装 `lerobot_rokae` 各 editable
package 的环境中：

```bash
cd /home/suhexi/rokae/lerobot_rokae
source /PATH/TO/LEROBOT_VENV/bin/activate
export PYTHONPATH=/home/suhexi/rokae/RLinf:${PYTHONPATH}
python /home/suhexi/rokae/RLinf/toolkits/rokae_hg_dagger/gateway.py \
  --config_path=/home/suhexi/rokae/RLinf/toolkits/rokae_hg_dagger/gateway_example.yaml
```

开放训练机到机器人电脑的 TCP `5560`，不要把该端口暴露到非受信网络。

键盘规则（焦点放在机器人电脑）：

- `S`：成功，reward=1，结束并保存该 episode。
- `F`：失败，reward=0，结束；`only_success=true` 会丢弃该 episode。
- `R`：立即结束并复位；同样不保存。
- `Esc`：软件安全停止并让训练端因连接中断而停下。硬件急停仍是独立的最高优先级保护。

SpaceMouse 任一轴越过 deadband 或按钮触发后，人工接管至少保持 0.5 秒。gateway
执行并回传的是 IK/processor 之后的最终 7 维关节目标，因此 HG-DAgger 保存的 expert
label 与 checkpoint 动作空间一致。

## 3. 启动 RLinf

编辑 `examples/embodiment/config/env/rokae_remote.yaml` 的 `ROBOT_PC_IP`，并将
`rokae_hg_dagger_pi05.yaml` 中四个 `/ABS/PATH/...` 替换为转换后的目录。然后在训练机：

```bash
cd /home/suhexi/rokae/RLinf
source /PATH/TO/RLINF_OPENPI_VENV/bin/activate
ray start --head --port=6379
bash examples/embodiment/run_realworld_async.sh rokae_hg_dagger_pi05
```

建议先把 `actor.optim.lr` 保持在 `1e-5`，用 3--5 个成功 episode 做冒烟测试。确认
`../results/.../online_lerobot/` 中出现 episode、`intervene_flag` 只标记人工帧、动作始终
是 7 维后，再进行长时间训练。

## 4. 一键流程脚本

也可以使用 `run_realworld_hg_dagger.sh` 串起复制 step 3000、转换、校验、Ray
和 RLinf 训练。脚本默认只从远端读取 checkpoint，不会写远端；checkpoint 转换使用
`OPENPI_PYTHON`，训练进程使用的 `RLINF_PYTHON` 必须同时能导入 RLinf、OpenPI
和 Ray（可以是同一个 OpenPI 虚拟环境）：

```bash
cd /home/suhexi/rokae/RLinf
OPENPI_PYTHON=/path/to/openpi/.venv/bin/python \
RLINF_PYTHON=/path/to/rlinf/.venv/bin/python \
ROBOT_PC_IP=ROBOT_PC_IP \
bash toolkits/rokae_hg_dagger/run_realworld_hg_dagger.sh prepare
```

服务器上使用已有的 OpenPI 训练环境时，推荐明确指定只读源码目录和统一输出目录：

```bash
OPENPI_ROOT=/vepfs-1/users/piaoweiyi/Projects/openpi \
OPENPI_PYTHON=/.venv/bin/python \
RLINF_PYTHON=/.venv/bin/python \
RUN_NAME=schaeffler3d_cr_orange_round_front_left_pi05_rtc_hgdagger_v1_20260824_001 \
ROBOT_PC_IP=ROBOT_PC_IP \
bash toolkits/rokae_hg_dagger/run_realworld_hg_dagger.sh prepare
```

`prepare` 完成 SSH/SCP、checkpoint 转换和配置校验。确认机器人电脑上的 gateway
已启动、SpaceMouse 和键盘焦点正常后，再运行：

```bash
SKIP_COPY=1 SKIP_CONVERT=1 \
ROBOT_PC_IP=ROBOT_PC_IP \
bash toolkits/rokae_hg_dagger/run_realworld_hg_dagger.sh train
```

### 非交互式服务器任务

如果调度系统只允许提交一个入口命令，使用包装脚本。`prepare` 在提交训练任务前
手动执行一次；它会在指定 run 目录生成 `jax_step3000/` 和
`pi05_step3000_torch/`。提交任务时只执行 `train`，不会再访问 checkpoint 服务器、
执行 SCP 或转换模型。

```bash
cd /vepfs-1/users/sunhaoxuan/RLinf

# 提交训练任务前执行一次，可观察日志并确认 checkpoint 校验通过。
ROBOT_PC_IP=192.168.1.20 \
bash toolkits/rokae_hg_dagger/run_server_hg_dagger.sh prepare

# 调度系统的非交互式入口命令。
ROBOT_PC_IP=192.168.1.20 \
bash toolkits/rokae_hg_dagger/run_server_hg_dagger.sh train
```

包装脚本默认使用新服务器镜像的 `/root/miniconda3/bin/python`，并固定只读
OpenPI 路径 `/vepfs-1/users/piaoweiyi/Projects/openpi`。如调度系统需要显式指定
地址，可使用 `GATEWAY_ADDRESS=tcp://192.168.1.20:5560`；它会覆盖由
`ROBOT_PC_IP` 推导出的地址。gateway 必须在机器人电脑上提前启动并保持运行。

若 gateway 与训练机是同一台机器，可以设置 `START_LOCAL_GATEWAY=1`（或自定义
`GATEWAY_CMD`）后运行 `all`；通常 gateway 在机器人电脑单独启动，因此训练端脚本
不会重复启动它，只通过 ZeroMQ 连接。脚本支持 `GATEWAY_ADDRESS=tcp://IP:5560`、
`RAY_HEAD=1` 或 `RAY_ADDRESS=head_ip:6379` 等环境变量，完整参数见：

```bash
bash toolkits/rokae_hg_dagger/run_realworld_hg_dagger.sh --help
```

## 上机前检查

1. 单独运行原有 SpaceMouse 遥操作，确认关节限位、夹爪方向和 reset trajectory。
2. gateway 启动后先发送全零以外的“当前关节保持”动作，而不是数学零关节动作。
3. 相机必须是 RGB、键名严格为 `external`/`wrist`，且方向与原始数据一致。
4. 检查转换目录的 norm stats SHA256；远端 step 3000 文件已核对为
   `80903fca49d191e756d09ef00b747474d3b4a157ba570f05c661c46e0226b21a`。
5. 首次闭环把机器人速度/加速度限制降到采集时水平以下，并让操作员手持急停。
