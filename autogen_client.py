import asyncio
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from exam.agent import get_agents
from exam.llm_provider import get_llm


class SimpleTokenCounter:
    def __init__(self):
        self.total = 0
        self.prompt = 0
        self.completion = 0

    def add(self, usage):
        if usage:

            p = getattr(usage, 'prompt_tokens', 0)
            c = getattr(usage, 'completion_tokens', 0)
            self.prompt += p
            self.completion += c
            self.total += (p + c)


async def main():
    exam_date = input("Please enter the exam date (e.g., 2025-06-05): ").strip()
    if not exam_date: exam_date = "2025-06-05"

    cost_counter = SimpleTokenCounter()

    team = SelectorGroupChat(
        get_agents(),
        model_client=get_llm(),
        termination_condition=TextMentionTermination("TERMINATE")
    )

    task = f"Start the exam assessment for date {exam_date}. First load exam AND checklists. Then assess the students."

    print(f"\n[AUTOGEN] Starting Assessment for {exam_date}...\n")

    async for message in team.run_stream(task=task):

        if hasattr(message, 'source') and hasattr(message, 'content'):
            print(f"\n[{message.source}]: {message.content}")

        if hasattr(message, 'models_usage'):
            cost_counter.add(message.models_usage)
            print(message.models_usage)

    print(f"Total Tokens:      {cost_counter.total}")
    print(f" - Prompt Tokens:  {cost_counter.prompt}")
    print(f" - Completion:     {cost_counter.completion}")


if __name__ == '__main__':
    asyncio.run(main())