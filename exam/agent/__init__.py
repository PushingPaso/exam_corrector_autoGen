from autogen_agentchat.agents import AssistantAgent


from exam.llm_provider import get_llm
from exam.mcp import ExamMCPServer

def get_agents():
    mcp = ExamMCPServer()
    UploaderAgent = AssistantAgent(
        name="Uploader File",
        model_client=get_llm(),
        tools=[mcp.load_checklist,mcp.load_exam_from_yaml_tool],
        system_message="Use tools to solve tasks.",
    )

    AssessorAgent = AssistantAgent(
        name="Uploader File",
        model_client=get_llm(),
        tools=[mcp.assess_student_exam],
        system_message="Use tools to solve tasks.",
    )


    return [UploaderAgent, AssessorAgent]



