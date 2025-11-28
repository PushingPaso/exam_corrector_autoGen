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


def analyze_framework_overhead(experiment_name="AutoGen_Exam_Assessment", debug=True):
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

            # Get inputs and outputs for better classification
            inputs = str(span.inputs).lower() if span.inputs else ""
            outputs = str(span.outputs).lower() if span.outputs else ""
            attributes = str(span.attributes).lower() if span.attributes else ""

            # More comprehensive orchestration detection
            orchestration_keywords = [
                "select", "speaker", "next", "role", "agent",
                "orchestrat", "coordinator", "manager", "system"
            ]

            is_orchestration = any(keyword in name or
                                   keyword in inputs or
                                   keyword in outputs or
                                   keyword in attributes
                                   for keyword in orchestration_keywords)

            if debug:
                print(f"LLM Call #{llm_count}:")
                print(f"  Name: {span.name}")
                print(f"  Duration: {duration_sec:.2f}s")
                print(f"  Classified as: {'ORCHESTRATION' if is_orchestration else 'CONVERSATION'}")
                if is_orchestration:
                    print(f"  Reason: Keywords found in name/inputs/outputs")
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
    print(f"Pure Conversation Time:  {conv_sec:.2f}s")
    print(f"Orchestration Overhead:  {over_sec:.2f}s")
    print(f"Total LLM Time:          {total_llm_sec:.2f}s")
    print(f"Framework Overhead Ratio: {overhead_ratio:.1f}%")
    print("-" * 50)