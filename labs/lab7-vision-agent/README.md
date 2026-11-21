# 实验七：Vision-Agents 实时视觉链路（A + B2）

## 实验内容

1. A：用 GetStream Vision-Agents 跑通实时视觉链路（Stream 传输 + Gemini Live 推理 + 本地 YOLO 姿态估计），并完成交互问答。
2. B2：控制变量对照，只更换姿态模型权重（yolo26n-pose 与 yolo26s-pose），比较处理帧率与回答质量。

## 环境

- Windows 11，Python 3.13.7，uv 0.13.9
- Vision-Agents 固定 commit：`902db86438aab03fb64aa7e6ea3afdcc867a3e80`
- 完整依赖见 `dependencies.txt`（uv pip freeze 导出，共 118 项）
- 关键依赖：torch 2.15.0+cpu、ultralytics 8.4.216、google-genai 2.34.0、getstream 4.3.0

## 准备

1. 克隆 Vision-Agents 到本地并切换到上面的 commit；
2. 将 `general_agent.py` 放入 `examples/02_golf_coach_example/`；
3. 把 `.env.example` 复制为 `.env`，填入 Stream 与 Gemini 的密钥；
4. 在示例目录执行 `uv sync --python 3.13`。

## 运行

```bash
# 基线（yolo26n-pose）
YOLO_MODEL=yolo26n-pose.pt uv run general_agent.py run

# 对照（yolo26s-pose）
YOLO_MODEL=yolo26s-pose.pt uv run general_agent.py run
```

Windows cmd 下用 `set "YOLO_MODEL=..."`。

## 数据

- `data/experiment_data.csv`：6 轮对照数据（两组各 3 轮）
- `logs/run_n1.log`：基线一次运行的脱敏日志
- 对照结果：fps 12.32 → 6.91（−44%），固定问题通过数 2.67/3 → 3.00/3

## 说明

1. 密钥只写在本地 `.env`，未提交到仓库；
2. 日志已脱敏，去掉了 api_key 与 token；
3. 演示视频较大，放在百度网盘（链接见仓库根目录 README）。
