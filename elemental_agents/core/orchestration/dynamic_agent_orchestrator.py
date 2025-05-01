"""
Agent orchestrator module. This module contains the main logic for orchestrating
the agents in the system. The orchestrator takes a planner, an agent, and a verifier
as input and runs them in sequence. This logic leads to decomposing the task into
sub-tasks, executing the sub-tasks with the agent, and verifying the final result
with the verifier.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from elemental_agents.core.agent.agent import Agent
from elemental_agents.core.agent_team.agent_team import AgentTeam
from elemental_agents.core.instruction.instruction import (
    BasicInstruction,
    ComposerInstruction,
    Instruction,
    ReplanningInstruction,
)
from elemental_agents.core.orchestration.orchestrator import Orchestrator
from elemental_agents.core.taskqueue.task import Status
from elemental_agents.observability.observer import observer
from elemental_agents.utils.exceptions import AgentException
from elemental_agents.utils.utils import extract_tag_content


class DynamicAgentOrchestrator(Orchestrator):
    """
    Main logic for orchestrating the agents in the system. The orchestrator
    takes a planner, an agent, and a verifier as input and runs them in sequence.
    This logic leads to decomposing the task into sub-tasks, executing the sub-tasks
    with the agent, and verifying the final result with the verifier.
    """

    def __init__(
        self,
        planner: Optional[Agent | AgentTeam] = None,
        plan_verifier: Optional[Agent | AgentTeam] = None,
        replanner: Optional[Agent | AgentTeam] = None,
        executor: Optional[Agent | AgentTeam] = None,
        verifier: Optional[Agent | AgentTeam] = None,
        composer: Optional[Agent | AgentTeam] = None,
    ) -> None:
        """
        Initialize the orchestrator with the planner, agent, and verifier executors.

        :param planner: The planner executor.
        :param plan_verifier: The plan verifier executor.
        :param agent: The agent executor.
        :param verifier: The verifier executor.
        """
        super().__init__()

        self._planner = planner
        self._plan_verifier = plan_verifier
        self._replanner = replanner
        self._executor = executor
        self._verifier = verifier
        self._composer = composer

    def run(
        self, instruction: str | List[str], input_session: str, restart: bool = False
    ) -> str:
        """
        Run the orchestrator with the specified instruction. The orchestrator
        generates a plan using the planner, creates a task queue with the plan,
        completes the tasks in the task queue using the agent, and verifies the
        final result using the verifier.

        :param instruction: The instruction to run the orchestrator with.
        :param input_session: The input session to run the orchestrator with.
        :param restart: A flag to indicate if the orchestrator is running in the
                        restart mode. In the restart mode, the orchestrator
                        skips the task queue creation and assumes it has been
                        set up elsewhere.
        :return: The final result from the verifier or the agent.
        """

        current_instruction = ""
        if isinstance(instruction, list):
            current_instruction = instruction[-1]
        else:
            current_instruction = instruction

        # -------------------------------------------------------------------- #
        # Planner
        # -------------------------------------------------------------------- #
        if (self._planner is not None) and (not restart):

            # Generate the plan
            logger.info(f"Running the planner with task: {instruction}")
            planner_response = self._planner.run(instruction, input_session)

            plan = extract_tag_content(planner_response, "JSON")
        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # Plan Verifier (only if planner is defined)
        # -------------------------------------------------------------------- #
        if (
            (self._plan_verifier is not None)
            and (self._planner is not None)
            and (not restart)
        ):

            # Verify the plan
            verifier_instruction = (
                f"<instruction>{current_instruction}</instruction>\n"
                f"<plan>{plan}</plan>"
            )
            verifier_response = self._plan_verifier.run(
                verifier_instruction, input_session
            )
            # Replace the plan
            plan = extract_tag_content(verifier_response, "JSON")
        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # Task Queue (only if planner is defined)
        # -------------------------------------------------------------------- #
        if (self._planner is not None) and (not restart):
            # Parse the plan
            parsed_plan: List[Dict[str, Any]] = []
            for p in plan:
                try:
                    t = p.replace("\\n", "").replace("\\t", "")
                    t = t.replace("\\\\", "").replace("\\", "")
                    parsed_plan.append(json.loads(t))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing task {p}: {e}")

            # Populate the task queue with the plan
            logger.info(f"Creating the task queue with the parsed plan: {parsed_plan}")
            self._taskqueue.create_task_queue(parsed_plan, current_instruction)

            # Print the initial task queue
            self._taskqueue.print_tasks()

            # Log the tasks and the task queue
            for _, task in self._taskqueue.get_all_tasks().items():
                observer.log_task(input_session, task)

        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # Agent execution with task queue (restart mode starts here)
        # -------------------------------------------------------------------- #
        if (self._executor is not None) and (self._planner is not None):

            # Complete the tasks in the task queue
            while True:

                (task, queue_status) = self._taskqueue.get_next_ready_tasks()

                if queue_status is Status.DONE:
                    break
                task_id = task.id

                # Retrieve memories relevant to the task
                memories: List[str] = []
                max_memories = self._config.long_memory_items
                if self._config.use_long_memory:
                    memories = self._memory.get(max_memories, task.description)

                # Create the instruction for the agent
                agent_instruction = Instruction(task, memories)

                logger.info(f"Running the agent with task: {task}")
                try:
                    agent_response = self._executor.run(
                        agent_instruction.render(), input_session
                    )

                except AgentException as e:
                    logger.error(f"Error running agent: {e}")
                    agent_response = f"Error running agent {e}"

                logger.info(f"Agent response: {agent_response}")

                # Update the task in the queue
                self._taskqueue.update_task(task_id, Status.DONE, agent_response)

                # Log the task
                observer.log_task(input_session, self._taskqueue.get_task(task_id))

                if self._config.use_long_memory:
                    self._memory.add(agent_response)

                # Replanning stage
                if self._replanner is not None:

                    replan_instruction = ReplanningInstruction(
                        taskqueue=self._taskqueue,
                        original_instruction=current_instruction,
                    )
                    logger.info(
                        f"Replanner with instruction: {replan_instruction.render()}"
                    )

                    try:
                        replanner_response = self._replanner.run(
                            replan_instruction.render(), input_session
                        )

                        revised_plan = extract_tag_content(replanner_response, "JSON")

                        parsed_revised_plan: List[Dict[str, Any]] = []
                        for p in revised_plan:
                            try:
                                t = p.replace("\\n", "").replace("\\t", "")
                                t = t.replace("\\\\", "").replace("\\", "")
                                parsed_revised_plan.append(json.loads(t))
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Revised plan -> Error parsing task {p}: {e}"
                                )

                        # Modify the task queue with the revised plan
                        logger.info(
                            f"Updating the task queue with the revised plan: {parsed_revised_plan}"
                        )
                        self._taskqueue.revise_task_queue(
                            parsed_revised_plan, current_instruction
                        )

                        # Print the initial task queue
                        self._taskqueue.print_tasks()

                        # Log the tasks and the task queue
                        for _, task in self._taskqueue.get_all_tasks().items():
                            observer.log_task(input_session, task)

                    except AgentException as e:
                        logger.error(f"Error running RePlanner: {e}")
                        replanner_response = f"Error running replanner {e}"

        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # Composer (only if planner and agent are defined)
        # -------------------------------------------------------------------- #
        if (
            (self._composer is not None)
            and (self._planner is not None)
            and (self._executor is not None)
        ):

            logger.info("Running the composer")
            composer_instruction = ComposerInstruction(
                self._taskqueue, current_instruction
            )
            composer_response = self._composer.run(
                composer_instruction.render(), input_session
            )
            agent_response = composer_response

        # -------------------------------------------------------------------- #
        # Verifier (only if planner and agent are defined)
        # -------------------------------------------------------------------- #
        if (
            (self._verifier is not None)
            and (self._planner is not None)
            and (self._executor is not None)
        ):

            # Verify the final result (last response from the agent)
            logger.info(f"Running the verifier with task: {agent_response}")
            full_instruction = (
                f"<request>{instruction}</request>\n"
                f"<response>{agent_response}</response>"
            )
            verifier_response = self._verifier.run(full_instruction, input_session)
            logger.info(f"Verifier response: {verifier_response}")

            # If the verifier response is false, return the verifier response
            if verifier_response.lower().strip() == "false":
                logger.error("Verification failed")
                return "Verification failed"
            else:
                logger.info("Verification passed")
                verifier_response = agent_response

            # Print the final task queue
            logger.info("Final state of the task queue:")
            self._taskqueue.print_tasks()

            return verifier_response

        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # Direct agent execution (no use of task queue)
        # -------------------------------------------------------------------- #
        if (self._executor is not None) and (self._planner is None):
            logger.info("No planner defined, running the agent directly")

            # Directly run instructions with the agent, no use of taskqueue
            logger.debug(f"Running the agent with task: {instruction}")

            if self._config.use_long_memory:

                max_memories = self._config.long_memory_items
                memories = self._memory.get(max_memories, current_instruction)
                inst = BasicInstruction(current_instruction, memories)

                # Render the instruction with memories
                current_instruction_with_memories = inst.render()

                # Replace the last instruction
                if isinstance(instruction, list):
                    instruction[-1] = current_instruction_with_memories
                else:
                    instruction = current_instruction_with_memories

            try:
                agent_response = self._executor.run(instruction, input_session)

            except AgentException as e:
                logger.error(f"Error running agent: {e}")
                agent_response = "Error running agent"

            if self._config.use_long_memory:
                self._memory.add(agent_response)

            logger.debug(f"Agent response: {agent_response}")
        # -------------------------------------------------------------------- #

        if self._executor is not None:
            return agent_response
        if self._planner is not None:
            return planner_response
        return "No agent response"

    def restart(
        self, instruction: str, input_session: str, previous_tasks: List[Dict[str, Any]]
    ) -> str:
        """
        Run the orchestrator with the set of previously created tasks. The
        orchestrator will skip the planning step and directly use the provided
        tasks to create the task queue. The orchestrator will then complete the
        tasks in the task queue using the agent and verify the final result
        using the verifier.

        :param instruction: The instruction to run the orchestrator with.
        :param input_session: The input session to run the orchestrator with.
        :param previous_tasks: The list of previously created tasks.
        :return: The final result from the the agent workflow.
        """

        # Populate the task queue with the instruction
        logger.info("Creating the task queue with the previous tasks")

        self._taskqueue.create_task_queue(
            tasks=previous_tasks, original_instruction=instruction, keep_ids=True
        )

        # Mark previously completed tasks as DONE and propagate
        # their results to the context of other tasks
        self._taskqueue.mark_tasks_with_results_as_done()
        self._taskqueue.fix_context_for_all_tasks()

        # Print the initial task queue
        self._taskqueue.print_tasks()

        # Run the orchestrator in the restart mode (jump to the agent execution)
        return self.run(instruction, input_session, restart=True)
