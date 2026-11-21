# 计算机视觉课程实验（2026 秋）

- 姓名：杨子亨
- 学号：2023090915021
- 学院：信息与软件工程学院

## 目录结构

```
├─ labs/
│  ├─ proj1-image-filtering/      # 实验一：图像滤波与混合图像
│  ├─ proj5-object-detection/     # 实验五：目标检测
│  └─ lab7-vision-agent/          # 实验七：Vision-Agents 实时视觉链路（A + B2）
├─ reports/
│  ├─ 实验报告-proj1.pdf
│  ├─ 实验报告-proj5.pdf
│  ├─ 实验报告-专题七.pdf
│  └─ 科技调研报告.pdf
├─ logs/
│  └─ run_n1.log                  # 专题七运行日志（已脱敏）
├─ data/
│  └─ experiment_data.csv         # B2 对照数据
├─ env/
│  └─ 环境与版本.txt
└─ video/
   └─ 演示视频（或百度网盘链接，见下）
```

## 实验七说明

- 框架：GetStream Vision-Agents，固定 commit `902db86438aab03fb64aa7e6ea3afdcc867a3e80`
- 模型：Gemini Live（实时语音+视觉）、YOLO26 pose（n / s 两组对照）
- 对照结果：fps 12.32 → 6.91（−44%），固定问题通过数 2.67/3 → 3.00/3
- 详细数据见 `data/experiment_data.csv` 与报告第 6 章

## 运行方式（实验七）

```bash
cd labs/lab7-vision-agent
uv sync --python 3.13
# 在 .env 中配置 STREAM_API_KEY / STREAM_API_SECRET / GEMINI_API_KEY
uv run general_agent.py run
```

`.env` 不在仓库中，运行前需自行配置。

## 演示视频

文件较大，放在百度网盘：`<在此填写网盘链接>`（提取码：`<填写>`）
