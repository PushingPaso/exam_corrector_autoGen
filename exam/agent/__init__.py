from autogen_agentchat.agents import AssistantAgent
from exam.llm_provider import get_llm
from exam.mcp import ExamMCPServer
import mlflow
from mlflow.entities import SpanType



@mlflow.trace(span_type=SpanType.AGENT)
def get_agents():
    mcp = ExamMCPServer()
    model = get_llm()
    UploaderAgent = AssistantAgent(
        name="uploader",
        model_client=model,
        tools=[mcp.load_exam_from_yaml_tool, mcp.load_checklist],
        system_message="""You are the Exam Data Manager.

        YOUR WORKFLOW (Follow strictly):
        1. Call `load_exam_from_yaml_tool`. 
           IMPORTANT: Provide ONLY the filenames (e.g., "se-2025-06-05-questions.yml"), DO NOT include paths like "static/".
           Expected filenames pattern: se-{DATE}-questions.yml, se-{DATE}-responses.yml, se-{DATE}-grades.yml.
        2. The output will contain 'question_ids'. Call `load_checklist` with these IDs.
        3. ONLY AFTER both tools success, output: "DATA READY", after every student exam is been evaluated output: "TERMINATE" .
        """
    )
    AssessorAgent = AssistantAgent(
        name="assessor",
        model_client=model,
        tools=[mcp.list_students, mcp.assess_students_batch],
        system_message="""You are the Exam Grader.

        CONDITION: Do NOT act until the uploader says "DATA READY".

        YOUR WORKFLOW:
        1. Call `list_students` to get the emails.
        2. Call `assess_students_batch` passing ALL emails at once and after 
        3. Say "TERMINATE" in chat when assess_students_batch methods ends.
        """
    )

    return [UploaderAgent, AssessorAgent]