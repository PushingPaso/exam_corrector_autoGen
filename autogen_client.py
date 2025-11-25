import asyncio
import os
from autogen_agentchat.teams import SelectorGroupChat  # <--- CAMBIO IMPORTANTE
from autogen_agentchat.conditions import TextMentionTermination

from exam.agent import get_agents
from exam.llm_provider import get_llm


async def main():
    exam_date = input("Please enter the exam date (e.g., 2025-06-05): ").strip()
    if not exam_date: exam_date = "2025-06-05"

    team = SelectorGroupChat(
        get_agents(),
        model_client=get_llm(),  # supervisor
        termination_condition=TextMentionTermination("TERMINATE")
    )

    # Prompt iniziale chiaro
    task = f"Start the exam assessment for date {exam_date}. First load exam AND checklists. Then assess the students."

    # Esegui in streaming per vedere i passaggi
    async for message in team.run_stream(task=task):
        if hasattr(message, 'source') and hasattr(message, 'content'):
            print(f"\n[{message.source}]: {message.content}")


if __name__ == '__main__':
    asyncio.run(main())