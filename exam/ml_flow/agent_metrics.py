"""
Advanced MLflow metrics tracking for multi-agent systems.
Provides sophisticated analysis of agent communication, performance, and behavior.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import mlflow


@dataclass
class AgentMessage:
    """Represents a single agent message with metadata."""
    timestamp: float
    source: str
    content: str
    token_count: int = 0
    message_type: str = "text"

    @property
    def latency_from_start(self) -> float:
        """Calculate latency from conversation start."""
        return self.timestamp


@dataclass
class AgentInteraction:
    """Represents an interaction between two agents."""
    from_agent: str
    to_agent: str
    timestamp: float
    message_length: int
    tokens_used: int = 0
    tool_calls: List[str] = field(default_factory=list)


class AgentMetricsTracker:
    """
    Comprehensive metrics tracker for multi-agent systems.
    Tracks communication patterns, performance, and agent behavior.
    """

    def __init__(self, experiment_name: str = "Agent_Communication_Analysis"):
        self.experiment_name = experiment_name
        self.start_time = None
        self.messages: List[AgentMessage] = []
        self.interactions: List[AgentInteraction] = []

        # Aggregate metrics
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "message_count": 0,
            "total_tokens": 0,
            "total_chars": 0,
            "tool_calls": [],
            "response_times": [],
            "first_message_time": None,
            "last_message_time": None
        })

        self.tool_usage: Dict[str, int] = defaultdict(int)
        self.conversation_turns: List[Dict[str, Any]] = []

    def start_tracking(self):
        """Initialize tracking with timestamp."""
        self.start_time = time.time()
        mlflow.log_param("tracking_start", datetime.now().isoformat())

    def log_message(self, source: str, content: str, token_count: int = 0,
                   message_type: str = "text", tool_name: Optional[str] = None):
        """
        Log a single agent message.

        Args:
            source: Name of the agent sending the message
            content: Message content
            token_count: Number of tokens in the message
            message_type: Type of message (text, tool_call, tool_result)
            tool_name: Name of tool if this is a tool call
        """
        if self.start_time is None:
            self.start_tracking()

        timestamp = time.time()
        relative_time = timestamp - self.start_time

        message = AgentMessage(
            timestamp=relative_time,
            source=source,
            content=content,
            token_count=token_count,
            message_type=message_type
        )
        self.messages.append(message)

        # Update agent stats
        stats = self.agent_stats[source]
        stats["message_count"] += 1
        stats["total_tokens"] += token_count
        stats["total_chars"] += len(content)

        if stats["first_message_time"] is None:
            stats["first_message_time"] = relative_time
        stats["last_message_time"] = relative_time

        if tool_name:
            stats["tool_calls"].append(tool_name)
            self.tool_usage[tool_name] += 1

    def log_interaction(self, from_agent: str, to_agent: str,
                       message_length: int, tokens: int = 0,
                       tool_calls: Optional[List[str]] = None):
        """
        Log an interaction between two agents.

        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            message_length: Length of message in characters
            tokens: Token count for this interaction
            tool_calls: List of tools called in this interaction
        """
        if self.start_time is None:
            self.start_tracking()

        interaction = AgentInteraction(
            from_agent=from_agent,
            to_agent=to_agent,
            timestamp=time.time() - self.start_time,
            message_length=message_length,
            tokens_used=tokens,
            tool_calls=tool_calls or []
        )
        self.interactions.append(interaction)

    def calculate_response_time(self, agent_name: str) -> float:
        """
        Calculate average response time for an agent.
        Response time = time between receiving a message and sending next message.
        """
        agent_messages = [m for m in self.messages if m.source == agent_name]
        if len(agent_messages) < 2:
            return 0.0

        response_times = []
        for i in range(1, len(agent_messages)):
            time_diff = agent_messages[i].timestamp - agent_messages[i-1].timestamp
            response_times.append(time_diff)

        return sum(response_times) / len(response_times) if response_times else 0.0

    def calculate_communication_graph(self) -> Dict[str, Dict[str, int]]:
        """
        Build a communication graph showing message flow between agents.

        Returns:
            Dict mapping from_agent -> to_agent -> message_count
        """
        graph = defaultdict(lambda: defaultdict(int))

        for i in range(len(self.messages) - 1):
            current = self.messages[i]
            next_msg = self.messages[i + 1]
            graph[current.source][next_msg.source] += 1

        return dict(graph)

    def calculate_agent_efficiency(self, agent_name: str) -> Dict[str, float]:
        """
        Calculate efficiency metrics for a specific agent.

        Returns:
            Dict with metrics like tokens_per_message, chars_per_token, etc.
        """
        stats = self.agent_stats[agent_name]

        if stats["message_count"] == 0:
            return {
                "tokens_per_message": 0.0,
                "chars_per_message": 0.0,
                "chars_per_token": 0.0,
                "tool_usage_rate": 0.0
            }

        return {
            "tokens_per_message": stats["total_tokens"] / stats["message_count"],
            "chars_per_message": stats["total_chars"] / stats["message_count"],
            "chars_per_token": stats["total_chars"] / max(stats["total_tokens"], 1),
            "tool_usage_rate": len(stats["tool_calls"]) / stats["message_count"]
        }

    def calculate_conversation_velocity(self) -> float:
        """
        Calculate messages per second across entire conversation.
        """
        if not self.messages or self.start_time is None:
            return 0.0

        total_time = self.messages[-1].timestamp
        return len(self.messages) / max(total_time, 1.0)

    def get_tool_usage_distribution(self) -> Dict[str, float]:
        """
        Get distribution of tool usage as percentages.
        """
        total = sum(self.tool_usage.values())
        if total == 0:
            return {}

        return {
            tool: (count / total) * 100
            for tool, count in self.tool_usage.items()
        }

    def calculate_idle_time(self) -> Dict[str, float]:
        """
        Calculate idle time (gaps between messages) for each agent.
        """
        idle_times = defaultdict(list)

        for agent in self.agent_stats.keys():
            agent_messages = sorted(
                [m for m in self.messages if m.source == agent],
                key=lambda x: x.timestamp
            )

            for i in range(1, len(agent_messages)):
                gap = agent_messages[i].timestamp - agent_messages[i-1].timestamp
                idle_times[agent].append(gap)

        return {
            agent: sum(gaps) / len(gaps) if gaps else 0.0
            for agent, gaps in idle_times.items()
        }

    def log_all_metrics_to_mlflow(self):
        """
        Log all calculated metrics to MLflow.
        """
        # Overall metrics
        mlflow.log_metric("total_messages", len(self.messages))
        mlflow.log_metric("total_interactions", len(self.interactions))
        mlflow.log_metric("conversation_velocity", self.calculate_conversation_velocity())
        mlflow.log_metric("total_agents", len(self.agent_stats))

        if self.messages:
            mlflow.log_metric("conversation_duration", self.messages[-1].timestamp)

        # Per-agent metrics
        for agent_name, stats in self.agent_stats.items():
            prefix = f"agent_{agent_name}"

            mlflow.log_metric(f"{prefix}_message_count", stats["message_count"])
            mlflow.log_metric(f"{prefix}_total_tokens", stats["total_tokens"])
            mlflow.log_metric(f"{prefix}_avg_response_time",
                            self.calculate_response_time(agent_name))

            efficiency = self.calculate_agent_efficiency(agent_name)
            for key, value in efficiency.items():
                mlflow.log_metric(f"{prefix}_{key}", value)

        # Tool usage metrics
        tool_dist = self.get_tool_usage_distribution()
        for tool, percentage in tool_dist.items():
            mlflow.log_metric(f"tool_{tool}_usage_pct", percentage)

        # Idle time metrics
        idle_times = self.calculate_idle_time()
        for agent, idle_time in idle_times.items():
            mlflow.log_metric(f"agent_{agent}_avg_idle_time", idle_time)

        # Communication graph
        comm_graph = self.calculate_communication_graph()
        mlflow.log_dict(comm_graph, "communication_graph.json")

    def generate_report(self) -> str:
        """
        Generate a human-readable report of all metrics.
        """
        lines = ["=" * 80]
        lines.append("AGENT COMMUNICATION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Messages: {len(self.messages)}")
        lines.append(f"Total Agents: {len(self.agent_stats)}")
        lines.append(f"Conversation Velocity: {self.calculate_conversation_velocity():.2f} msg/sec")

        if self.messages:
            lines.append(f"Duration: {self.messages[-1].timestamp:.2f} seconds")
        lines.append("")

        # Per-agent details
        lines.append("AGENT DETAILS")
        lines.append("-" * 80)

        for agent_name in sorted(self.agent_stats.keys()):
            stats = self.agent_stats[agent_name]
            efficiency = self.calculate_agent_efficiency(agent_name)

            lines.append(f"\n{agent_name}:")
            lines.append(f"  Messages: {stats['message_count']}")
            lines.append(f"  Tokens: {stats['total_tokens']}")
            lines.append(f"  Avg Response Time: {self.calculate_response_time(agent_name):.2f}s")
            lines.append(f"  Tokens/Message: {efficiency['tokens_per_message']:.1f}")
            lines.append(f"  Tool Usage Rate: {efficiency['tool_usage_rate']:.2%}")

            if stats['tool_calls']:
                lines.append(f"  Tools Used: {', '.join(set(stats['tool_calls']))}")

        # Tool usage
        if self.tool_usage:
            lines.append("\nTOOL USAGE DISTRIBUTION")
            lines.append("-" * 80)
            tool_dist = self.get_tool_usage_distribution()
            for tool, pct in sorted(tool_dist.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {tool}: {pct:.1f}% ({self.tool_usage[tool]} calls)")

        # Communication patterns
        lines.append("\nCOMMUNICATION PATTERNS")
        lines.append("-" * 80)
        comm_graph = self.calculate_communication_graph()
        for from_agent, targets in comm_graph.items():
            for to_agent, count in targets.items():
                lines.append(f"  {from_agent} -> {to_agent}: {count} messages")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_report(self, filepath: str = "agent_metrics_report.txt"):
        """Save the report to a file and log to MLflow."""
        report = self.generate_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        mlflow.log_artifact(filepath)
        return filepath