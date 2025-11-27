import asyncio
import time
import mlflow
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
    # 1. ENABLE NATIVE AUTOLOGGING (Captures Traces automatically)
    # Requires mlflow >= 2.16.0
    if hasattr(mlflow, 'autogen'):
        mlflow.autogen.autolog()
    else:
        print("WARNING: mlflow.autogen not found. Please upgrade mlflow.")

    exam_date = input("Please enter the exam date (e.g., 2025-06-05): ").strip()
    if not exam_date:
        exam_date = "2025-06-05"

    cost_counter = SimpleTokenCounter()
    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("AutoGen_Exam_Assessment")

    with mlflow.start_run():
        mlflow.log_param("framework", "AutoGen")
        mlflow.log_param("exam_date", exam_date)

        llm_client = get_llm()
        mlflow.log_param("model_name", "gpt-4o")

        team = SelectorGroupChat(
            get_agents(),
            model_client=llm_client,
            termination_condition=TextMentionTermination("TERMINATE")
        )

        task = f"Start the exam assessment for date {exam_date}. First load exam AND checklists. Then assess the students."

        print(f"\n[AUTOGEN] Starting Assessment for {exam_date}...\n")

        start_time = time.time()

        async for message in team.run_stream(task=task):
            if hasattr(message, 'source') and hasattr(message, 'content'):
                print(f"\n[{message.source}]: {message.content}")

            # Capture usage for the thesis table
            if hasattr(message, 'models_usage'):
                cost_counter.add(message.models_usage)

        end_time = time.time()
        duration = end_time - start_time

        # 3. PRINT CONSOLE REPORT
        print(f"Total Tokens: {cost_counter.total}")
        print(f"Prompt Tokens: {cost_counter.prompt}")
        print(f"Completion Tokens: {cost_counter.completion}")
        print(f"Duration (seconds): {duration:.2f}")

        # 4. LOG AGGREGATE METRICS TO MLFLOW
        mlflow.log_metric("total_tokens", cost_counter.total)
        mlflow.log_metric("prompt_tokens", cost_counter.prompt)
        mlflow.log_metric("completion_tokens", cost_counter.completion)
        mlflow.log_metric("duration_seconds", duration)


if __name__ == '__main__':
    asyncio.run(main())