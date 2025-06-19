"""
Functions to setup the workflow with the planner, agents, and verifier for the
single or multi-agent orchestration. The workflow is initialized with
the configuration file/string that specifies all stages and their parameters.
"""

from typing import Any, Dict, List

from loguru import logger

from elemental_agents.core.agent.agent import Agent
from elemental_agents.core.agent.generic_agent import GenericAgent
from elemental_agents.core.agent.simple_agent import SimpleAgent
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.composer_agent import ComposerAgentLogic
from elemental_agents.core.agent_logic.conv_planreact_agent import (
    ConvPlanReActAgentLogic,
)
from elemental_agents.core.agent_logic.plan_verifier_agent import PlanVerifierAgentLogic
from elemental_agents.core.agent_logic.planner_agent import PlannerAgentLogic
from elemental_agents.core.agent_logic.planreact_agent import PlanReActAgentLogic
from elemental_agents.core.agent_logic.react_agent import ReActAgentLogic
from elemental_agents.core.agent_logic.replanner_agent import ReplannerAgentLogic
from elemental_agents.core.agent_logic.simple_agent import SimpleAgentLogic
from elemental_agents.core.agent_logic.verifier_agent import VerifierAgentLogic
from elemental_agents.core.agent_team.agent_team import AgentTeam
from elemental_agents.core.agent_team.generic_agent_team import GenericAgentTeam
from elemental_agents.core.orchestration.dynamic_agent_orchestrator import (
    DynamicAgentOrchestrator,
)
from elemental_agents.core.selector.agent_selector_factory import AgentSelectorFactory
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm_factory import LLMFactory


class Workflow:
    """
    Workflow class to setup the planner, agents, and verifier for the single or
    multi-agent orchestration. The workflow is initialized with the
    configuration file that specifies the planner, agents, and verifier and
    their parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the workflow with the specified configuration.

        :param config: The configuration dictionary for the workflow.
        """

        self._config = config

        self._planner: Agent | AgentTeam | None = None
        self._plan_verifier: Agent | AgentTeam | None = None
        self._replanner: Agent | AgentTeam | None = None
        self._agents: Agent | AgentTeam | None = None
        self._verifier: Agent | AgentTeam | None = None
        self._composer: Agent | AgentTeam | None = None

        logger.debug(f"Workflow initialized with config: {self._config}")

        self._llm_factory = LLMFactory()
        self._orchestrator = self.setup_workflow()

        logger.debug("Workflow setup completed.")

    def setup_stage_agents(self, stage_name: str) -> Agent | AgentTeam:
        """
        Setup the agents based on the configuration. The agents can be single or
        multiple agents based on the configuration. The agents are setup with
        the specified LLM, context, and toolbox.

        :param stage_name: The stage name to setup the agents (e.g. planner,
            executor, verifier).
        :return: The agent executor or unit executor based on the configuration.
        """

        agent_composition: Agent | AgentTeam | None = None

        try:
            if len(self._config[stage_name]) > 1:
                agent_composition = self.setup_stage_agent_team(stage_name)
            else:
                agent_composition = self.setup_stage_agent(stage_name)
        except TypeError as e:
            logger.error(f"{stage_name} section is expected to be a list. {e}")

        return agent_composition

    def setup_stage_agent(self, stage_name: str) -> Agent:
        """
        Setup the agent executor with the specified LLM, context, and toolbox.

        :param stage_name: The stage name to setup the agents (e.g. planner,
            executor, verifier).
        :return: The agent executor.
        """

        stage_config = self._config[stage_name][0]

        # Create the LLM for the agent
        if "llm" in stage_config:
            agent_llm_type = stage_config["llm"]

            llm_parameters = ModelParameters(
                temperature=stage_config["temperature"],
                stop=(stage_config["stopWords"]).split(","),
                max_tokens=stage_config["maxTokens"],
                frequency_penalty=stage_config["frequencyPenalty"],
                presence_penalty=stage_config["presencePenalty"],
                top_p=stage_config["topP"],
            )

            llm = self._llm_factory.create(agent_llm_type, llm_parameters)
        else:
            llm = self._llm_factory.create()

        # Set agent specific memory settings
        if "memory" in stage_config:
            memory_capacity = stage_config["memory"]
        else:
            memory_capacity = -1

        # Agent name and persona
        context = AgentContext(
            agent_name=stage_config["name"], agent_persona=stage_config["persona"]
        )

        # Register tools for the agent
        tools = stage_config["tools"]
        toolbox = ToolBox()

        for tool in tools:
            toolbox.register_tool_by_name(tool)

        # Prompt strategy
        agent_type = stage_config["type"]

        agent: (
            SimpleAgentLogic
            | ReActAgentLogic
            | PlanReActAgentLogic
            | PlannerAgentLogic
            | PlanVerifierAgentLogic
            | VerifierAgentLogic
            | ReplannerAgentLogic
            | None
        ) = None

        agent_prompt_template = None
        if "template" in stage_config:
            agent_prompt_template = stage_config["template"]

        match agent_type:
            case "Simple":
                agent = SimpleAgentLogic(llm, context, agent_prompt_template)
            case "ReAct":
                agent = ReActAgentLogic(llm, context, toolbox, agent_prompt_template)
            case "PlanReAct":
                agent = PlanReActAgentLogic(
                    llm, context, toolbox, agent_prompt_template
                )
            case "PlanVerifier":
                agent = PlanVerifierAgentLogic(llm, context, agent_prompt_template)
            case "Planner":
                agent = PlannerAgentLogic(llm, context, agent_prompt_template)
            case "RePlanner":
                agent = ReplannerAgentLogic(llm, context, agent_prompt_template)
            case "Verifier":
                agent = VerifierAgentLogic(llm, context, agent_prompt_template)
            case "Composer":
                agent = SimpleAgentLogic(llm, context, agent_prompt_template)
            case _:
                logger.error(f"Agent type {agent_type} is not supported.")
                agent = None

        executor: SimpleAgent | GenericAgent | None = None

        # Create the individual agent executor based on the agent type
        if agent_type in [
            "Simple",
            "Planner",
            "RePlanner",
            "PlanVerifier",
            "Verifier",
            "Composer",
        ]:
            executor = SimpleAgent(agent=agent, short_memory_capacity=memory_capacity)
        else:
            executor = GenericAgent(
                agent_logic=agent,
                short_memory_capacity=memory_capacity,
                toolbox=toolbox,
                termination_sequence="<result>",
            )

        return executor

    def setup_stage_agent_team(self, stage_name: str) -> AgentTeam:
        """
        Setup the agent team with the specified agents, LLM, context, and
        toolbox. The AgentTeam object is used for multi-agent orchestration.

        :param stage_name: The stage name to setup the agents (e.g. planner,
            executor, verifier).
        :return: The unit executor.
        """

        logger.info(f"Setting up the AgentTeam for {stage_name}")

        # Get the agent selector type from the configuration
        agent_selector_type = ""
        lead_agent = ""

        if "unit" in self._config:
            if stage_name in self._config["unit"]:
                logger.info(
                    f"Unit configuration found. {self._config['unit'][stage_name]}"
                )

                if "selector" in self._config["unit"][stage_name]:
                    agent_selector_type = self._config["unit"][stage_name]["selector"]

                if "leadAgent" in self._config["unit"][stage_name]:
                    lead_agent = self._config["unit"][stage_name]["leadAgent"]

        agent_selector_factory = AgentSelectorFactory()
        agent_selector = agent_selector_factory.create(agent_selector_type, lead_agent)

        # Initialize the agent unit
        agent_unit = GenericAgentTeam(selector=agent_selector)

        stage_config = self._config[stage_name]

        # Create agents and register them with their executors in the unit
        for agent_config in stage_config:

            # Create the LLM for the agent, if specified in the configuration
            # choose the specified LLM type otherwise, use the default LLM
            if "llm" in agent_config:
                agent_llm_type = agent_config["llm"]

                llm_parameters = ModelParameters(
                    temperature=agent_config["temperature"],
                    stop=(agent_config["stopWords"]).split(","),
                    max_tokens=agent_config["maxTokens"],
                    frequency_penalty=agent_config["frequencyPenalty"],
                    presence_penalty=agent_config["presencePenalty"],
                    top_p=agent_config["topP"],
                )

                llm = self._llm_factory.create(agent_llm_type, llm_parameters)
            else:
                llm = self._llm_factory.create()

            if "memory" in agent_config:
                memory_capacity = agent_config["memory"]
            else:
                memory_capacity = -1

            # Agent name and persona
            context = AgentContext(
                agent_name=agent_config["name"], agent_persona=agent_config["persona"]
            )

            # Register tools for the agent
            tools = agent_config["tools"]
            toolbox = ToolBox()

            for tool in tools:
                toolbox.register_tool_by_name(tool)

            agent_type = agent_config["type"]

            agent: (
                SimpleAgentLogic
                | ReActAgentLogic
                | PlanReActAgentLogic
                | PlannerAgentLogic
                | PlanVerifierAgentLogic
                | VerifierAgentLogic
                | ReplannerAgentLogic
                | ConvPlanReActAgentLogic
                | ComposerAgentLogic
                | None
            ) = None

            agent_prompt_template = None
            if "template" in agent_config:
                agent_prompt_template = agent_config["template"]

            # Create the agent based on the agent type
            match agent_type:
                case "Simple":
                    agent = SimpleAgentLogic(llm, context, agent_prompt_template)
                case "ReAct":
                    agent = ReActAgentLogic(
                        llm, context, toolbox, agent_prompt_template
                    )
                case "PlanReAct":
                    agent = PlanReActAgentLogic(
                        llm, context, toolbox, agent_prompt_template
                    )
                case "ConvPlanReAct":
                    agent = ConvPlanReActAgentLogic(
                        llm, context, toolbox, agent_prompt_template
                    )
                case "PlanVerifier":
                    agent = PlanVerifierAgentLogic(llm, context, agent_prompt_template)
                case "Planner":
                    agent = PlannerAgentLogic(llm, context, agent_prompt_template)
                case "RePlanner":
                    agent = ReplannerAgentLogic(llm, context, agent_prompt_template)
                case "Verifier":
                    agent = VerifierAgentLogic(llm, context, agent_prompt_template)
                case "Composer":
                    agent = ComposerAgentLogic(llm, context, agent_prompt_template)
                case _:
                    logger.error(f"Agent type {agent_type} is not supported.")
                    agent = None

            # Executor for the agent
            agent_executor: SimpleAgent | GenericAgent | None = None

            # Create the individual agent executor based on the agent type
            if agent_type in [
                "Simple",
                "Planner",
                "PlanVerifier",
                "Verifier",
                "RePlanner",
                "Composer",
            ]:
                agent_executor = SimpleAgent(
                    agent=agent, short_memory_capacity=memory_capacity
                )
            else:
                agent_executor = GenericAgent(
                    agent_logic=agent,
                    short_memory_capacity=memory_capacity,
                    toolbox=toolbox,
                    termination_sequence="<result>",
                )

            # Register the agent with the agent unit
            agent_unit.register_agent(
                agent_name=agent_config["name"],
                executor=agent_executor,
                agent_type=agent_type,
            )

        # Update the prompt strategy for the conversational agents
        agent_unit.update_conversational_strategies()

        return agent_unit

    def setup_workflow(self) -> DynamicAgentOrchestrator:
        """
        Setup the workflow with the planner, agents, and verifier executors.

        :return: The workflow orchestrator.
        """

        for stage in self._config["workflow"]:

            stage_agents = self.setup_stage_agents(stage)

            match stage:
                case "planner":
                    self._planner = stage_agents
                case "rePlanner":
                    self._replanner = stage_agents
                case "planVerifier":
                    self._plan_verifier = stage_agents
                case "executor":
                    self._agents = stage_agents
                case "verifier":
                    self._verifier = stage_agents
                case "composer":
                    self._composer = stage_agents
                case _:
                    logger.error(f"Stage {stage} is not supported.")
                    raise ValueError(f"Stage {stage} is not supported.")

        # Setup the orchestrator based on the number of agents
        orchestrator: DynamicAgentOrchestrator | None = None

        orchestrator = DynamicAgentOrchestrator(
            planner=self._planner,
            plan_verifier=self._plan_verifier,
            replanner=self._replanner,
            executor=self._agents,  # type: ignore
            verifier=self._verifier,
            composer=self._composer,
        )

        return orchestrator

    def run(
        self,
        input_instruction: str | List[str] | None = None,
        input_session: str | None = None,
    ) -> str:
        """
        Run the workflow orchestrator with the specified instruction.

        :param input_instruction: The instruction for Agents (list in case of
            continuation of the conversation)
        :param input_session: The session ID
        :return: The final result from the verifier.
        """

        if input_instruction is None:
            if "instruction" in self._config:
                instruction = self._config["instruction"]
            else:
                logger.error("No instruction provided.")
                raise ValueError("No instruction provided.")
        else:
            instruction = input_instruction

        result = self._orchestrator.run(instruction, input_session)

        return result

    def restart(
        self,
        input_instruction: str | None = None,
        input_session: str | None = None,
        previous_tasks: List[Dict[str, Any]] | None = None,
    ) -> str:
        """
        Restart the workflow orchestrator with the specified instruction.

        :param input_instruction: The instruction for Agents
        :param input_session: The session ID
        :param previous_tasks: The previous tasks from the session to be resumed.
        :return: The final result from the the agent workflow.
        """

        if input_instruction is None:
            if "instruction" in self._config:
                instruction = self._config["instruction"]
            else:
                logger.error("No instruction provided.")
                raise ValueError("No instruction provided.")
        else:
            instruction = input_instruction

        result = ""
        try:
            result = self._orchestrator.restart(
                instruction, input_session, previous_tasks
            )
        except Exception as e:
            message = f"Error occurred during the restart: {e}"
            logger.error(message)
            result = message

        return result
