from autogen_agentchat.agents import AssistantAgent


from exam.llm_provider import get_llm
from exam.mcp import ExamMCPServer

def get_agents():
    mcp = ExamMCPServer()
    UploaderAgent = AssistantAgent(
        name="uploader",
        model_client=get_llm(),
        tools=[mcp.load_checklist,mcp.load_exam_from_yaml_tool],
        system_message="Your task it to first upload exam data and after the checkilist of the question, you will find anl the file in static/se-exams/ at the data given."
                       "Use tools to solve tasks.",
    )

    AssessorAgent = AssistantAgent(
        name="assessor",
        model_client=get_llm(),
        tools=[mcp.assess_student_exam],
        system_message="Your task it to assess the students exam using the relative mail"
                       "Use tools to solve tasks.",
    )


    return [UploaderAgent, AssessorAgent]



