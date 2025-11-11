import asyncio
import os
from autogen_agentchat.teams import RoundRobinGroupChat

from exam.agent import get_agents


async def main():
    exam_date = input("Please enter the exam date: es(2025-06-05).....")
    team = RoundRobinGroupChat(get_agents())
    result = await team.run(task=f"Upload and asses the exam in date {exam_date}")
    print(result)


if __name__ == '__main__':
    asyncio.run(main())
