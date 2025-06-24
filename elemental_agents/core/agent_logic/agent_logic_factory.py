"""
Factory for creating agent logic instances.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import AgentLogic
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.llm import LLM


class AgentLogicFactory:
    """
    Factory class for creating agent logic instances.
    """

    @staticmethod
    def create_agent_logic(
        logic_type: str,
        llm: LLM,
        context: AgentContext,
        template: Optional[str] = None,
        toolbox: Optional[ToolBox] = None,
    ) -> AgentLogic:
        """
        Creates an instance of the specified agent logic type.

        :param logic_type: The type of agent logic to create.
        :param llm: The LLM instance to use for the agent logic.
        :param context: The context for the agent logic, including name and persona.
        :param template: Optional template string to use for the agent logic.
        :param toolbox: Optional toolbox instance to use for the agent logic.
        :return: An instance of the specified agent logic.
        """

        supported_agent_logic_types = [
            "simple",
            "planner",
            "planverifier",
            "replanner",
            "composer",
            "verifier",
            "react",
            "planreact",
            "convplanreact",
        ]

        agent_logic: (
            SimpleAgentLogic
            | ComposerAgentLogic
            | VerifierAgentLogic
            | PlannerAgentLogic
            | PlanVerifierAgentLogic
            | ReplannerAgentLogic
            | ReActAgentLogic
            | PlanReActAgentLogic
            | ConvPlanReActAgentLogic
        ) = None

        match logic_type:
            case "simple":
                from elemental_agents.core.agent_logic.simple_agent import (
                    SimpleAgentLogic,
                )

                agent_logic = SimpleAgentLogic(
                    model=llm, context=context, template=template
                )
                return agent_logic
            case "composer":
                from elemental_agents.core.agent_logic.composer_agent import (
                    ComposerAgentLogic,
                )

                agent_logic = ComposerAgentLogic(
                    model=llm, context=context, template=template
                )
                return agent_logic
            case "verifier":
                from elemental_agents.core.agent_logic.verifier_agent import (
                    VerifierAgentLogic,
                )

                agent_logic = VerifierAgentLogic(
                    model=llm, context=context, template=template
                )
                return agent_logic
            case "planner":
                from elemental_agents.core.agent_logic.planner_agent import (
                    PlannerAgentLogic,
                )

                agent_logic = PlannerAgentLogic(
                    model=llm, context=context, template=template
                )
                return agent_logic
            case "planverifier":
                from elemental_agents.core.agent_logic.plan_verifier_agent import (
                    PlanVerifierAgentLogic,
                )

                agent_logic = PlanVerifierAgentLogic(
                    model=llm, context=context, template=template
                )
                return agent_logic
            case "replanner":
                from elemental_agents.core.agent_logic.replanner_agent import (
                    ReplannerAgentLogic,
                )

                agent_logic = ReplannerAgentLogic(
                    model=llm, context=context, template=template
                )
                return agent_logic
            case "react":
                from elemental_agents.core.agent_logic.react_agent import (
                    ReActAgentLogic,
                )

                agent_logic = ReActAgentLogic(
                    model=llm, context=context, toolbox=toolbox, template=template
                )
                return agent_logic
            case "planreact":
                from elemental_agents.core.agent_logic.planreact_agent import (
                    PlanReActAgentLogic,
                )

                agent_logic = PlanReActAgentLogic(
                    model=llm, context=context, toolbox=toolbox, template=template
                )
                return agent_logic
            case "convplanreact":
                from elemental_agents.core.agent_logic.conv_planreact_agent import (
                    ConvPlanReActAgentLogic,
                )

                agent_logic = ConvPlanReActAgentLogic(
                    model=llm, context=context, toolbox=toolbox, template=template
                )
                return agent_logic
            case _:
                logger.error(
                    f"Unknown agent logic type: {logic_type}. "
                    f"Supported agent logic types are: {supported_agent_logic_types}."
                )
                raise ValueError(
                    f"Unknown agent logic type: {logic_type}. "
                    f"Supported agent logic types are: {supported_agent_logic_types}."
                )
