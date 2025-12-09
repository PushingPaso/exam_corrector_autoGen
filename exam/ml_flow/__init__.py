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
