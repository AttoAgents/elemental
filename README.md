<div align="center"> 
<img src="https://attoagents.io/images/elemental-main-logo-light.png" width="256" alt="Elemental"> 
</div>

![GitHub License](https://img.shields.io/github/license/AttoAgents/elemental?style=for-the-badge)
---

Elemental is a general-purpose, multi-agent framework for automating tasks using AI agents composed in teams for conversation-based collaboration. Agents can use language models from various providers (including small models in local environment), allowing more flexible environment choice and isolation of work from external infrastructure if needed. Leading LLMs are also supported. Agents are equipped with tools that allow them to interact with external systems (internal and provided with MCP servers). Elemental allows for various configurations including simple assistant, isolated agent, multi-agent teams and workflows or multi-agent teams. The agent configuration is dynamic, with the ability to plan work, and replan tasks during execution.

## Features

- Multi-agent task execution.
- Custom language model per agent (including different inference engines and size of the model - direct support for Ollama, Llama.cpp, OpenAI and compatible APIs, Anthropic).
- Simple model selection per agent e.g. `ollama|gemma3` or `openai|gpt-4.1-mini`.
- Customizable templates with variables in Jinja format.
- Default dynamic orchestrator with dynamic planning, execution, re-planning, composition and verification steps.
- Simple command line interface with agent configuration provided by YAML file.
- Tool execution with extendable interface to provide native tools executable with any language model.
- Reasoning and conversational agent prompt strategies.
- MCP Tools with complete toolset or individual tool level selection.

## Getting started 

## Install

Install with:

```
pip install elemental-agents
```

### Install from sources

```
git clone git@github.com:AttoAgents/elemental.git
cd elemental
poetry build
pip install dist/elemental_agents-*.whl
```

### `.env` file

Example of `.env` file for Elemental projects is 

```sh
openai_api_key="<OPENAI API KEY HERE>"
openai_streaming=False
openai_max_tokens=10000
default_engine="ollama"
custom_max_tokens=2000
google_search_api_key="<GOOGLE SEARCH API HERE>"
google_cse_id="<GOOGLE CSE ID HERE>"
google_search_timeout=5
wikipedia_user_agent="<Agents/1.0 YOUR EMAIL HERE>"
observer_destination="screen"
mcpServers='{"Github": {"command": "npx", "args": ["-y","@modelcontextprotocol/server-github"], "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR GITHUB TOKEN>"}}}'
```

Elemental will by default use the `.env` file from the current directory, unless an environmental variable `ATTO_ENV_FILE` which specifies the path to another configuration file.

## Examples 

Below we show examples on how to use `elemental_agents` framework to programmatically setup simple assistant, agents, agent teams, use tools and utilize external MCP servers. All examples shown here are also available in the `examples` directory.

### Simple assistant

The simplest agent configuration in Elemental is setup of an simple assistant that does not have ability to execute tools and serves as an interface to the language model. The assistant however is conversation aware and includes `ShortMemory` that stores the conversation history.

```python
from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory

if __name__ == "__main__":

    TASK = "Why is the sky blue?"
    SESSION = "TestSession"

    factory = AgentFactory()
    assistant = factory.create(
        agent_name="AssistantAgent",
        agent_persona="Simple always helpful assistant",
        agent_type="simple",
        llm_model="ollama|gemma3",
    )

    result = assistant.run(task=TASK, input_session=SESSION)
    logger.info(f"Result: {result}")

```

In this case we utilize [Ollama](https://ollama.com) with Gemma3 model ([See example 1](examples/01-simple-assistant/example.py)).

### ReAct agent

More complex and complete agents can be created with utilizing one of the iterative reasoning prompt strategies like ReAct. In this case agent will be able to utilize tools by executing actions and brining the results as observations. In the example below ([See example 2](examples/02-ReAct-agent/example.py)) we equip the agent with several tools. 

```python
from loguru import logger

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
    logger.info(f"Result: {result}")

```

The task demonstrates the need to use the `Calculator` tool. In this example we utilize language model provided by [OpenAI](https://openai.com) API. 

> [!NOTE]  
> Elemental does not rely on the function calling ability of particular language model and handles the definition of actions (which select tools and their parameter) with the prompt strategy of an agent. 

### ReAct agent with internal planning - PlanReAct

Similarly to ReAct agent we can define more complex prompt strategy that includes internal planning. By selecting `agent_type="PlanReAct"` we can create an agent that augments ReAct strategy by internal planning ([See example 3](examples/03-PlanReAct-agent/example.py))

```python
from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory

if __name__ == "__main__":

    TASK = "Calculate the sum of 5 and 3."
    SESSION = "Test Session"

    factory = AgentFactory()
    assistant = factory.create(
        agent_name="AssistantAgent",
        agent_persona="You are a helpful assistant.",
        agent_type="PlanReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=["Calculator", "CurrentTime", "NoAction"],
    )
    result = assistant.run(task=TASK, input_session=SESSION)
    logger.info(f"Result: {result}")

```

### Conversational agent team 

A team of agents that are meant to work together may be defined by first creating the individual agents and then creating agent team with `GenericAgentTeam` class. To enable the conversational character of the agents they need to be created with `agent_type="ConvPlanReAct"`. This enables the conversational character and awareness of the team with prompt strategy ([See example 4](examples/04-agent-team/example.py))

```python
from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory
from elemental_agents.core.agent_team.generic_agent_team import GenericAgentTeam
from elemental_agents.core.selector.agent_selector_factory import AgentSelectorFactory

if __name__ == "__main__":

    factory = AgentFactory()
    agent1 = factory.create(
        agent_name="AssistantAgent",
        agent_persona="You are a helpful assistant.",
        agent_type="ConvPlanReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=["Calculator", "CurrentTime", "NoAction"],
    )
    agent2 = factory.create(
        agent_name="ProgrammerAgent",
        agent_persona="You are a helpful programmer.",
        agent_type="ConvPlanReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=["Calculator", "CurrentTime", "NoAction"],
    )

    selector_factory = AgentSelectorFactory()
    agent_selector = selector_factory.create(
        selector_name="conversational", lead_agent="AssistantAgent"
    )
    agent_team = GenericAgentTeam(selector=agent_selector)
    agent_team.register_agent("AssistantAgent", agent1, "ConvPlanReAct")
    agent_team.register_agent("ProgrammerAgent", agent2, "ConvPlanReAct")

    result = agent_team.run(
        task="What is the color of sky on Mars?", input_session="Example Session"
    )
    logger.info(f"Result: {result}")
```

The above task does not require (or potentially employ the conversation).  


### Orchestrated team of agents - external planning and task queue 

While a single agent may be used with internal planning prompt strategy like `PlanReAct`, the planning process may be done using a specialized planning agent. In this case agent creates the plan and populates a task queue. This process is orchestrated with flexible `DynamicAgentOrchestrator` class and may also include more steps including replanning done during the execution. 

The example below includes two simple agents to illustrate the process ([See example 5](examples/05-orchestrated-agents/example.py))

```python
from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory
from elemental_agents.core.orchestration.dynamic_agent_orchestrator import (
    DynamicAgentOrchestrator,
)

if __name__ == "__main__":

    factory = AgentFactory()

    planner_agent = factory.create(
        agent_name="PlannerAgent",
        agent_persona="",
        agent_type="planner",
        llm_model="openai|gpt-4.1-mini",
    )
    executor_agent = factory.create(
        agent_name="ExecutorAgent",
        agent_persona="You are an expert software engineer.",
        agent_type="ReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=[
                "Calculator",
                "CurrentTime",
                "NoAction",
                "ReadFiles",
                "WriteFile",
                "ListFiles"
            ],
    )

    orchestrator = DynamicAgentOrchestrator(planner=planner_agent, executor=executor_agent)

    result = orchestrator.run(
        instruction="Create FastAPI backend for a TODO application.",
        input_session="Example Session"
    )
    logger.info(f"Result: {result}")

```

The above example utilizes two steps in the workflow that `DynamicAgentOrchestrator` manages. The complete list includes `planner`, `plan_verifier`, `replanner`, `executor`, `verifier`, and `composer`.

### Definition using YAML files

The agent configuration may be provided as YAML file. Elemental includes `Workflow` and `Driver` classes that will set up all necessary objects (i.e. agents/agent teams) for `DynamicAgentOrchestrator` to run. 

Example configuration may be ([See example 6](examples/06-yaml-config-file/config-example.yaml))

```yaml
workflowName: ModelTest
workflow:
  - executor
executor:
  - name: Assistant
    type: Simple
    persona: >-
      You are expert researcher and great communicator of complex topics using
      simple terms. You always give comprehensive and extensive responses that
      consider the task at hand.
    tools: []
    llm: ollama|qwen3:14b
    temperature: 0
    frequencyPenalty: 0
    presencePenalty: 0
    topP: 1
    maxTokens: 2000
    stopWords: <PAUSE>, STOP
    template: >
      {{ agent_persona }}

      Follow user's instruction. Do this on a stepwise basis and double-check
      each step, one at a time. Use markdown in your response for more readable
      format. 
```

This YAML configuration may be used by incorporating `Driver` class programmatically or by using `elemental_agents.main.main` tool that serves as a very simple user interface.

The configuration provided above may be run with 

```
python -m elemental_agents.main.main --config example.yaml --instruction "Why is the sky blue?" 
```

In this case `--instruction` option augments this configuration by the user provided task.

### Model Context Protocol (MCP) Server tools

To use MCP Servers in Elemental one needs to define them in the configuration file using `mcpConfig` variable, e.g.

```sh
mcpServers='{"Github": {"command": "npx", "args": ["-y","@modelcontextprotocol/server-github"], "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR GITHUB TOKEN>"}}}'
```
The above value of `mcpConfig` variable adds the Github MCP server. More than one server may be defined in a similar fashion by adding additional entries in the JSON blob. 
A tool from an MCP server may be then added to the agent seamlessly with `MCP|server_name|tool_name` syntax. In the example below ([See example 7](examples/07-mcp-tools-single-tool/example.py)) we add `search_repositories` tool from Github MCP server defined above as `MCP|Github|search_repositories`.

```python
from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory

if __name__ == "__main__":

    TASK = "Search Github repositories for REST API creation in Python."
    SESSION = "Test Session"

    factory = AgentFactory()
    assistant = factory.create(
        agent_name="AssistantAgent",
        agent_persona="You are a helpful assistant.",
        agent_type="ReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=["Calculator", "CurrentTime", "NoAction", "MCP|Github|search_repositories"],
    )
    result = assistant.run(task=TASK, input_session=SESSION)
    logger.info(f"Result: {result}")
```

To make all tools provided by an MCP server available to the agent use `MCP|server_name|*` as a tool name. This will query the tools and register all of them. The example above may be modified by changing `MCP|Github|search_repositories` to `MCP|Github|*` ([See example 8](examples/08-mcp-tools-all-tools/example.py)).


## Contact
The project is developed and maintained by AttoAgents, for information please contact us at <info@attoagents.io>.
