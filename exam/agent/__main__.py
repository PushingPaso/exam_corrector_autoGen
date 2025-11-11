import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from exam.llm_provider import get_llm
from exam.mcp import ExamMCPServer

async def main():
    mcp = ExamMCPServer()

    agent = AssistantAgent(
        name="assistant",
        model_client=get_llm(),
        tools=[mcp.load_checklist,mcp.load_exam_from_yaml_tool],
        system_message="Use tools to solve tasks.",
    )

    result = await agent.run(task="You have to upload exam data from yaml file and chcklist.")
    print(result.messages)

if __name__ == "__main__":
    asyncio.run(main())