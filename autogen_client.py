"""
Enhanced AutoGen client with sophisticated metrics tracking.
This is an example showing how to integrate the agent_metrics module.
"""

import asyncio
import time
import mlflow
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from exam.agent import get_agents
from exam.llm_provider import get_llm
from exam.ml_flow import SimpleTokenCounter, calculate_overhead


async def main():
    if hasattr(mlflow, 'autogen'):
        mlflow.autogen.autolog()

    exam_date = input("Please enter the exam date (e.g., 2025-06-05): ").strip()
    if not exam_date:
        exam_date = "2025-06-05"

    # Initialize counters
    cost_counter = SimpleTokenCounter()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("AutoGen_Exam_Assessment")

    with mlflow.start_run()as run:
        mlflow.log_param("framework", "AutoGen")
        mlflow.log_param("exam_date", exam_date)
        mlflow.log_param("metrics_tracking_enabled", True)

        llm_client = get_llm()
        mlflow.log_param("model_name", "gpt-4o")

        team = SelectorGroupChat(
            get_agents(),
            model_client=llm_client,
            termination_condition=TextMentionTermination("TERMINATE")
        )

        task = f"Start the exam assessment for date {exam_date}. First load exam AND checklists calling the uploader agent. Then assess the students with the assessor agent."

        print(f"\n[AUTOGEN] Starting Assessment for {exam_date}...\n")

        start_time = time.time()

        # Start metrics tracking
        message_count = 0

        async for message in team.run_stream(task=task):
            message_count += 1

            if hasattr(message, 'source') and hasattr(message, 'content'):
                print(f"\n[{message.source}]: {message.content}")

                if hasattr(message, 'models_usage'):
                    cost_counter.add(message.models_usage)
                    print(message.models_usage)


        end_time = time.time()
        duration = end_time - start_time

        # Log basic metrics (as before)
        print(f"\nTotal Tokens: {cost_counter.total}")
        print(f"Prompt Tokens: {cost_counter.prompt}")
        print(f"Completion Tokens: {cost_counter.completion}")
        print(f"Duration (seconds): {duration:.2f}")

        mlflow.log_metric("total_tokens", cost_counter.total)
        mlflow.log_metric("prompt_tokens", cost_counter.prompt)
        mlflow.log_metric("completion_tokens", cost_counter.completion)
        mlflow.log_metric("duration_seconds", duration)

        calculate_overhead(run.info.run_id, duration)


if __name__ == '__main__':
    asyncio.run(main())
