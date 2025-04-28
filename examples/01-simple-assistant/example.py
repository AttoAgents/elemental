from elemental_agents.core.agent.agent_factory import AgentFactory

factory = AgentFactory()
assistant = factory.create("AssistantAgent", "Simple", "ollama|gemma3")
result = assistant.run("Why is the sky blue?")
