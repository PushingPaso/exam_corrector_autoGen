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
from exam.ml_flow import SimpleTokenCounter, analyze_framework_overhead
from exam.ml_flow.agent_metrics import AgentMetricsTracker


async def main():
    if hasattr(mlflow, 'autogen'):
        mlflow.autogen.autolog()

    exam_date = input("Please enter the exam date (e.g., 2025-06-05): ").strip()
    if not exam_date:
        exam_date = "2025-06-05"

    # Initialize counters
    cost_counter = SimpleTokenCounter()

    # Initialize agent metrics tracker
    metrics_tracker = AgentMetricsTracker(experiment_name="AutoGen_Exam_Assessment")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("AutoGen_Exam_Assessment")

    with mlflow.start_run():
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

        task = f"Start the exam assessment for date {exam_date}. First load exam AND checklists. Then assess the students."

        print(f"\n[AUTOGEN] Starting Assessment for {exam_date}...\n")

        start_time = time.time()

        # Start metrics tracking
        metrics_tracker.start_tracking()

        message_count = 0
        previous_agent = None

        async for message in team.run_stream(task=task):
            message_count += 1

            if hasattr(message, 'source') and hasattr(message, 'content'):
                print(f"\n[{message.source}]: {message.content}")

                # Calculate tokens for this message
                token_count = 0
                if hasattr(message, 'models_usage'):
                    cost_counter.add(message.models_usage)
                    if hasattr(message.models_usage, 'completion_tokens'):
                        token_count = message.models_usage.completion_tokens

                # Log message to metrics tracker
                metrics_tracker.log_message(
                    source=message.source,
                    content=message.content,
                    token_count=token_count,
                    message_type="text"
                )

                # Log interaction if there was a previous agent
                if previous_agent and previous_agent != message.source:
                    metrics_tracker.log_interaction(
                        from_agent=previous_agent,
                        to_agent=message.source,
                        message_length=len(message.content),
                        tokens=token_count
                    )

                previous_agent = message.source

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

        # Log all agent metrics to MLflow
        print("\n[METRICS] Computing advanced agent metrics...")
        metrics_tracker.log_all_metrics_to_mlflow()

        # Generate and save report
        print("\n[METRICS] Generating analysis report...")
        report_path = metrics_tracker.save_report(f"agent_metrics_{exam_date}.txt")
        print(f"[METRICS] Report saved to: {report_path}")

        # Print summary to console
        print("\n" + metrics_tracker.generate_report())

        # Log additional derived metrics
        conversation_velocity = metrics_tracker.calculate_conversation_velocity()
        mlflow.log_metric("messages_per_second", conversation_velocity)

        # Log communication efficiency
        comm_graph = metrics_tracker.calculate_communication_graph()
        total_transitions = sum(
            sum(targets.values()) for targets in comm_graph.values()
        )
        mlflow.log_metric("total_agent_transitions", total_transitions)

    # Wait before analysis
    time.sleep(3)

    # Perform framework overhead analysis
    print("\n[ANALYSIS] Analyzing framework overhead...")
    analyze_framework_overhead("AutoGen_Exam_Assessment")

    # Additional analysis options
    print("\n[ANALYSIS] You can run additional analyses:")
    print("  - from exam.ml_flow import compare_agent_performance")
    print("  - from exam.ml_flow import export_metrics_to_csv")
    print("  - from exam.ml_flow import create_experiment_comparison")


async def run_comparative_analysis():
    """
    Run a comparative analysis across multiple exam dates.
    Useful for understanding how agent performance varies.
    """
    from exam.ml_flow import create_experiment_comparison, compare_agent_performance

    print("\n[COMPARATIVE ANALYSIS] Starting...")

    # Compare different runs
    create_experiment_comparison([
        "AutoGen_Exam_Assessment",
        # Add other experiment names here if you have them
    ])

    # Compare specific agents
    compare_agent_performance(
        "AutoGen_Exam_Assessment",
        agent_names=["uploader", "assessor"]
    )


async def export_analysis():
    """
    Export all metrics for external analysis (e.g., in Excel, Python notebooks).
    """
    from exam.ml_flow import export_metrics_to_csv

    print("\n[EXPORT] Exporting metrics...")
    export_metrics_to_csv(
        "AutoGen_Exam_Assessment",
        output_file=f"metrics_export_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )


if __name__ == '__main__':
    asyncio.run(main())
