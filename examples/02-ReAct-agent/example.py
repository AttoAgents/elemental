"""
Example of ReAct agent usage with a simple task.
"""

from elemental_agents.core.agent.agent_factory import AgentFactory

if __name__ == "__main__":

    TASK = "Calculate the sum of 5 and 3."
    SESSION = "Test Session"

    factory = AgentFactory()
    assistant = factory.create(
        agent_name="AssistantAgent",
        agent_persona="You are a helpful assistant.",
        agent_type="ReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=["Calculator", "CurrentTime", "NoAction"],
    )
    result = assistant.run(task=TASK, input_session=SESSION)
