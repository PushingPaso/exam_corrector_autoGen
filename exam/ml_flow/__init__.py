import mlflow


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


def analyze_framework_overhead(experiment_name, debug=True):
    """
    Downloads the last trace from MLflow and calculates the time spent
    in pure conversation versus orchestration overhead.
    """
    print("\nSOPHISTICATED METRICS ANALYSIS (Trace Mining)")
    print("=" * 50)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Experiment not found.")
        return

    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        max_results=1,
        order_by=["timestamp DESC"]
    )

    if traces.empty:
        print("No traces found yet. Wait a few seconds and try again.")
        return

    trace_row = traces.iloc[0]
    request_id = trace_row['trace_id']

    trace = mlflow.get_trace(request_id)
    spans = trace.data.spans

    conversation_time_ns = 0
    overhead_time_ns = 0

    print(f"Analyzing Trace ID: {trace.info.trace_id}")
    print(f"Total spans found: {len(spans)}\n")

    llm_count = 0
    for span in spans:
        duration = span.end_time_ns - span.start_time_ns
        name = span.name.lower()
        span_type = span.span_type

        is_llm_call = span_type == "LLM"

        if is_llm_call:
            llm_count += 1
            duration_sec = duration / 1e9

            # Get inputs for classification
            inputs = str(span.inputs).lower() if span.inputs else ""

            # CRITICAL FIX: AutoGen's selector overhead is specifically about choosing next speaker
            # Tool calls (load_exam, assess_students, etc.) are NOT overhead - they're actual work!

            # Orchestration = selecting which agent speaks next
            is_selector_call = (
                    "selectorgroupchat" in name or
                    "selector" in name or
                    ("select" in inputs and "speaker" in inputs) or
                    ("next" in inputs and "agent" in inputs)
            )

            # Tool execution is PRODUCTIVE work, not overhead
            is_tool_call = (
                    "tool" in name or
                    "load_exam" in inputs or
                    "load_checklist" in inputs or
                    "assess" in inputs or
                    "list_students" in inputs
            )

            # Only selector calls are overhead, tool calls are conversation
            is_orchestration = is_selector_call and not is_tool_call

            if debug:
                print(f"LLM Call #{llm_count}:")
                print(f"  Name: {span.name}")
                print(f"  Duration: {duration_sec:.2f}m")
                print(f"  Is Selector: {is_selector_call}")
                print(f"  Is Tool Call: {is_tool_call}")
                print(f"  Classified as: {'ORCHESTRATION' if is_orchestration else 'CONVERSATION'}")
                if len(inputs) > 0:
                    print(f"  Input preview: {inputs[:150]}...")
                print()

            if is_orchestration:
                overhead_time_ns += duration
            else:
                conversation_time_ns += duration

    conv_sec = conversation_time_ns / 1e9
    over_sec = overhead_time_ns / 1e9
    total_llm_sec = conv_sec + over_sec

    if total_llm_sec == 0:
        print("No LLM calls detected in trace.")
        return

    overhead_ratio = (over_sec / total_llm_sec) * 100

    print("=" * 50)
    print(f"Total LLM Calls: {llm_count}")
    print(f"Pure Conversation Time:  {conv_sec:.2f}m")
    print(f"Orchestration Overhead:  {over_sec:.2f}m")
    print(f"Total LLM Time:          {total_llm_sec:.2f}m")
    print(f"Framework Overhead Ratio: {overhead_ratio:.1f}%")
    print("-" * 50)