"""通用视觉助手：能看见摄像头画面，正常识别并回应，不做高尔夫教练人设。

跑法：
    uv run general_agent.py run

B2 对照时只改环境变量，不改代码（控制变量）：
    $env:YOLO_MODEL="yolo26n-pose.pt"
    $env:YOLO_MODEL="yolo26s-pose.pt"
"""

import logging
import os

from dotenv import load_dotenv

from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream, ultralytics

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = os.environ.get("YOLO_MODEL", "yolo26n-pose.pt")

INSTRUCTIONS = """
你是一个友好的视觉助手。你能看到用户的摄像头画面，也能听到用户说话。

规则：
1. 开场只打一句招呼，不要长篇自我介绍。
2. 之后用户问什么，就根据你看到的画面回答，1-2 句，具体、简短。
3. 如果画面里看不清，就直接说看不清，不要编造。
4. 用户说中文就用中文回答，说英文就用英文回答。
5. 不要提自己是 AI 模型，也不要解释系统配置。

你可以描述：人在做什么动作、姿态是否标准、画面里有什么物体、光线与构图如何。
"""


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI assistant"),
        instructions=INSTRUCTIONS,
        llm=gemini.Realtime(fps=3),
        processors=[ultralytics.YOLOPoseProcessor(model_path=MODEL_PATH)],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response(text="Say hi in one short sentence, then wait for my questions.")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
