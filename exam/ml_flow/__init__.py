"""
Enhanced MLflow utilities with agent metrics integration.
"""

import mlflow
import pandas as pd
from typing import Optional


class SimpleTokenCounter:
    """Simple token counter for tracking LLM usage."""

    def __init__(self):
        self.total = 0
        self.prompt = 0
        self.completion = 0

    def add(self, models_usage):
        """Add usage from AutoGen models_usage object."""
        if hasattr(models_usage, 'prompt_tokens'):
            self.prompt += models_usage.prompt_tokens
        if hasattr(models_usage, 'completion_tokens'):
            self.completion += models_usage.completion_tokens

        self.total = self.prompt + self.completion


def analyze_framework_overhead(experiment_name: str):
    """
    Analyze framework overhead by comparing runs within an experiment.

    Args:
        experiment_name: Name of the MLflow experiment to analyze
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"[ANALYSIS] Experiment '{experiment_name}' not found")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        if runs.empty:
            print(f"[ANALYSIS] No runs found in experiment '{experiment_name}'")
            return

        print("\n" + "=" * 80)
        print(f"FRAMEWORK OVERHEAD ANALYSIS: {experiment_name}")
        print("=" * 80)

        # Display basic stats
        print(f"\nTotal Runs: {len(runs)}")
        print(f"\nRecent Runs Summary:")
        print("-" * 80)

        for idx, row in runs.head(5).iterrows():
            run_id = row['run_id'][:8]
            duration = row.get('metrics.duration_seconds', 0)
            total_tokens = row.get('metrics.total_tokens', 0)

            print(f"\nRun {run_id}:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Total Tokens: {int(total_tokens)}")

            # Agent-specific metrics
            agent_metrics = [col for col in runs.columns if col.startswith('metrics.agent_')]
            if agent_metrics:
                print(f"  Agent Metrics Found: {len(agent_metrics)}")
                for metric in sorted(agent_metrics)[:5]:  # Show first 5
                    value = row.get(metric, 0)
                    metric_name = metric.replace('metrics.', '')
                    print(f"    {metric_name}: {value:.2f}")

        # Statistical analysis if we have multiple runs
        if len(runs) > 1:
            print("\n" + "-" * 80)
            print("STATISTICAL SUMMARY")
            print("-" * 80)

            numeric_metrics = ['metrics.duration_seconds', 'metrics.total_tokens',
                               'metrics.prompt_tokens', 'metrics.completion_tokens']

            for metric in numeric_metrics:
                if metric in runs.columns:
                    values = runs[metric].dropna()
                    if not values.empty:
                        print(f"\n{metric.replace('metrics.', '')}:")
                        print(f"  Mean: {values.mean():.2f}")
                        print(f"  Std: {values.std():.2f}")
                        print(f"  Min: {values.min():.2f}")
                        print(f"  Max: {values.max():.2f}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"[ANALYSIS ERROR] {str(e)}")


def compare_agent_performance(experiment_name: str, agent_names: Optional[list] = None):
    """
    Compare performance metrics across different agents.

    Args:
        experiment_name: Name of the MLflow experiment
        agent_names: Optional list of agent names to compare
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"[COMPARISON] Experiment '{experiment_name}' not found")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        if runs.empty:
            print(f"[COMPARISON] No runs found")
            return

        # Get latest run
        latest_run = runs.iloc[0]

        print("\n" + "=" * 80)
        print(f"AGENT PERFORMANCE COMPARISON")
        print("=" * 80)

        # Extract agent metrics
        agent_metrics = {}
        for col in runs.columns:
            if col.startswith('metrics.agent_'):
                parts = col.replace('metrics.agent_', '').split('_', 1)
                if len(parts) == 2:
                    agent_name, metric_name = parts

                    if agent_names is None or agent_name in agent_names:
                        if agent_name not in agent_metrics:
                            agent_metrics[agent_name] = {}
                        agent_metrics[agent_name][metric_name] = latest_run.get(col, 0)

        if not agent_metrics:
            print("No agent metrics found in the latest run")
            return

        # Display comparison
        for agent_name in sorted(agent_metrics.keys()):
            metrics = agent_metrics[agent_name]
            print(f"\n{agent_name.upper()}:")
            print("-" * 40)

            for metric_name in sorted(metrics.keys()):
                value = metrics[metric_name]
                print(f"  {metric_name}: {value:.2f}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"[COMPARISON ERROR] {str(e)}")


def export_metrics_to_csv(experiment_name: str, output_file: str = "metrics_export.csv"):
    """
    Export all metrics from an experiment to CSV for external analysis.

    Args:
        experiment_name: Name of the MLflow experiment
        output_file: Path to output CSV file
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"[EXPORT] Experiment '{experiment_name}' not found")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        if runs.empty:
            print(f"[EXPORT] No runs found")
            return

        # Select relevant columns
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        essential_cols = ['run_id', 'start_time', 'end_time', 'status']

        export_cols = essential_cols + param_cols + metric_cols
        export_cols = [col for col in export_cols if col in runs.columns]

        export_df = runs[export_cols].copy()
        export_df.to_csv(output_file, index=False)

        print(f"[EXPORT] Metrics exported to {output_file}")
        print(f"  Rows: {len(export_df)}")
        print(f"  Columns: {len(export_df.columns)}")

        return output_file

    except Exception as e:
        print(f"[EXPORT ERROR] {str(e)}")


def log_conversation_flow(messages: list, artifact_name: str = "conversation_flow.json"):
    """
    Log conversation flow as an MLflow artifact.

    Args:
        messages: List of messages with source, content, timestamp
        artifact_name: Name of the artifact file
    """
    import json
    from pathlib import Path

    try:
        flow_data = {
            "total_messages": len(messages),
            "messages": [
                {
                    "index": i,
                    "source": msg.get("source", "unknown"),
                    "timestamp": msg.get("timestamp", 0),
                    "length": len(msg.get("content", ""))
                }
                for i, msg in enumerate(messages)
            ]
        }

        temp_file = Path(artifact_name)
        with open(temp_file, 'w') as f:
            json.dump(flow_data, f, indent=2)

        mlflow.log_artifact(str(temp_file))
        temp_file.unlink()  # Clean up

        print(f"[FLOW] Logged conversation flow: {len(messages)} messages")

    except Exception as e:
        print(f"[FLOW ERROR] {str(e)}")


def create_experiment_comparison(experiment_names: list):
    """
    Create a comparison report across multiple experiments.

    Args:
        experiment_names: List of experiment names to compare
    """
    print("\n" + "=" * 80)
    print("MULTI-EXPERIMENT COMPARISON")
    print("=" * 80)

    results = {}

    for exp_name in experiment_names:
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if not experiment:
                print(f"\n[WARNING] Experiment '{exp_name}' not found")
                continue

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )

            if runs.empty:
                continue

            latest = runs.iloc[0]
            results[exp_name] = {
                "duration": latest.get('metrics.duration_seconds', 0),
                "total_tokens": latest.get('metrics.total_tokens', 0),
                "total_messages": latest.get('metrics.total_messages', 0),
                "conversation_velocity": latest.get('metrics.conversation_velocity', 0)
            }

        except Exception as e:
            print(f"[ERROR] Failed to load {exp_name}: {str(e)}")

    if not results:
        print("\nNo experiment data found")
        return

    # Display comparison
    print("\n" + "-" * 80)
    print(f"{'Experiment':<40} {'Duration':<12} {'Tokens':<12} {'Messages':<12}")
    print("-" * 80)

    for exp_name, metrics in results.items():
        print(f"{exp_name:<40} "
              f"{metrics['duration']:<12.2f} "
              f"{int(metrics['total_tokens']):<12} "
              f"{int(metrics['total_messages']):<12}")

    print("=" * 80)