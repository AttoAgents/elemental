from elemental_agents.core.agent.agent_factory import AgentFactory

factory = AgentFactory()
assistant = factory.create(
    agent_name="AssistantAgent", agent_type="ReAct", llm="openai|gpt-4.1-mini"
)
result = assistant.run("Why is the sky blue?")
