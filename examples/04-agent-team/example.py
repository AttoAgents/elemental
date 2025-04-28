from elemental_agents.core.agent.agent_factory import AgentFactory
from elemental_agents.core.agent_team.agent_team import AgentTeam

factory = AgentFactory()

assistant = factory.create(
    agent_name="AssistantAgent",
    agent_type="ConvPlanReAct",
    agent_persona="Act as helpful assistant.",
    llm="openai|gpt-4.1-mini",
)
programmer = factory.create(
    agent_name="AssistantAgent",
    agent_type="ConvPlanReAct",
    agent_persona="Skillful programmer.",
    llm="ollama|gemma3",
)

agent_team = AgentTeam(selector=ConversationalSelector(lead_agent="AssistantAgent"))
agent_team.register_agent(assistant)
agent_team.register_agent(programmer)

agent_team.run(task="What is the color of sky on Mars?")
