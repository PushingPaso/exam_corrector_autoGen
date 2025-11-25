from autogen_agentchat.agents import AssistantAgent
from exam.llm_provider import get_llm
from exam.mcp import ExamMCPServer


def get_agents():
    mcp = ExamMCPServer()
    model = get_llm()

    # ISTRUZIONI RIGIDE PER L'UPLOADER
    # Deve fare due cose PRIMA di dare l'ok all'assessor.
    UploaderAgent = AssistantAgent(
        name="uploader",
        model_client=model,
        tools=[mcp.load_exam_from_yaml_tool,mcp.load_checklist],
        system_message="""You are the Exam Data Manager.

        YOUR WORKFLOW (Follow strictly):
        1. Call `load_exam_from_yaml_tool` with the provided date you will find the data at static/se-exams/se->{exam_date}-questions,static/se-exams/se->{exam_date}-responses, static/se-exams/se->{exam_date}-grades.
        2. The output will contain 'question_ids'. You MUST immediately call `load_checklist` with these IDs.
        3. ONLY AFTER both tools have run successfully, output a message: "DATA READY. Students to assess: [list of emails]".

        DO NOT STOP after the first tool. You must run both.
        """
    )

    # ISTRUZIONI RIGIDE PER L'ASSESSOR
    # Deve aspettare il segnale "DATA READY".
    AssessorAgent = AssistantAgent(
        name="assessor",
        model_client=model,
        tools=[mcp.list_students,mcp.assess_students_batch],
        system_message="""You are the Exam Grader.

        CONDITION: Do NOT act until the uploader has said "DATA READY" and loaded the checklists.

        YOUR WORKFLOW:
        1. Look for the list of student emails given by the tool .
        2. Call `assess_students_batch` with the provided emails.
        3. Once all assessments are done, output "TERMINATE".
        """
    )

    return [UploaderAgent, AssessorAgent]