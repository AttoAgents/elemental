"""
Class representing a template for prompts.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger


class PromptTemplate(ABC):
    """
    Abstract class for prompt templates. Templates will be rendered using the
    dictionary context that includes all the variables needed for the template.
    Two types of templates are supported: file templates and string templates.
    """

    def __init__(self, prompt_context: Dict[str, Any]) -> None:
        """
        Constructor for the PromptTemplate class. The context is a dictionary
        that includes all the variables needed for the template. This is an
        abstract class and should not be instantiated directly.

        :param context: Dictionary with all the variables needed for the
            template
        """

        self._context = prompt_context

    @abstractmethod
    def render(self) -> str:
        """
        Abstract method to render the template. All variables in the template
        will be replaced by the values in the context dictionary.
        """


class FileTemplate(PromptTemplate):
    """
    File template class. This class is based on Jinja2 templates. The template
    file should be in the templates directory and the file name should be passed
    as an argument to the constructor.
    """

    def __init__(self, prompt_context: Dict[str, str], file_name: str) -> None:
        """
        Constructor for the FileTemplate class. The context is a dictionary that
        includes all the variables needed for the template. The file name is the
        name of the template file that should be in the templates directory.

        :param context: Dictionary with all the variables needed for the
            template
        :param file_name: Name of the template file
        """

        super().__init__(prompt_context)

        self._file_name = file_name

        # Template files directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(current_dir, "../templates")

        self._file_loader = FileSystemLoader(templates_dir)
        self._env = Environment(loader=self._file_loader)
        self._template = self._env.get_template(self._file_name)

    def render(self) -> str:
        """
        Render the template. All variables in the template will be replaced by
        the values in the context dictionary.

        :return: The final prompt after rendering the template
        """

        final_prompt = self._template.render(self._context)

        return final_prompt


class StringTemplate(PromptTemplate):
    """
    String template class. This class is based on Python string formatting. The
    template string should be passed as an argument to the constructor.
    """

    def __init__(self, prompt_context: Dict[str, str], template: str) -> None:
        """
        Constructor for the StringTemplate class. The context is a dictionary
        that includes all the variables needed for the template. The template is
        a string that should include the variables to be replaced by the values
        in the context dictionary.

        :param context: Dictionary with all the variables needed for the
            template
        :param template: Template string with variables to be replaced
        """

        super().__init__(prompt_context)

        self._template = Template(template)

    def render(self) -> str:
        """
        Render the template. All variables in the template will be replaced by
        the values in the context dictionary.

        :return: The final prompt after rendering the template
        """
        # logger.debug(f"Context {self._context}")
        # logger.debug(f"Template {self._template}")

        final_prompt = self._template.render(**self._context)

        return final_prompt


if __name__ == "__main__":

    context = {
        "agent_name": "ResearchAgent",
        "agent_persona": "Researcher always following scientific method",
    }

    CHECK = """This is a test template for ResearchAgent.
               Researcher always following scientific method."""
    logger.info(CHECK)

    # File template
    file_template = FileTemplate(prompt_context=context, file_name="test.template")
    final_file_prompt = file_template.render()
    logger.info(final_file_prompt)

    # String template
    TEMPLATE = "This is a test template for {agent_name}. {agent_persona}."
    string_template = StringTemplate(prompt_context=context, template=TEMPLATE)
    final_string_prompt = string_template.render()
    logger.info(final_string_prompt)

    TEMPLATE = """
You are a planning assistant responsible for breaking down user requests into logical steps. First, extract key terms from the user's request, then decompose the request into a series of smaller, actionable steps. Each step should either directly contribute to the final result or be necessary for the completion of subsequent steps, such as comparisons, aggregations, or the use of intermediate results.

For questions, break them down into simpler sub-questions that lead to the final answer.

Provide your response in two sections: <thoughts> and <plan>.

<thoughts>: In this section, outline your reasoning, explain your approach, and ensure the steps make sense. Review your plan to ensure correctness. Provide the thoughs in <thoughs>...</thoughs> tags.
<plan>: In this section, break down the tasks in JSON format, each task in <JSON>...</JSON> tag. Provide the complete plan in <plan>...</plan> tags.

<JSON>
{
"id": (unique ID),
"description": (clear task instruction including needed values),
"dependencies": (list of task IDs this task depends on, if any)
}
</JSON>

Ensure each task can be completed independently with ALL the necessary details! List dependencies where needed. Do not reference other task names in the description.

Example: 
<thoughts>Explanation and reasoning.</thoughts> 
<plan> 
<JSON> { "id": 1, "description": "Description of Task 1", "dependencies": [] } </JSON> 
<JSON> { "id": 2, "description": "Description of Task 2", "dependencies": [1] } </JSON>
<JSON> { "id": 3, "description": "Complete the user's request.", "dependencies": [1,2] } </JSON>
</plan>

Review each step to ensure description is sufficient to carry it out without knowing other tasks. Make sure all XML tags are closed!

Let’s begin!

"""
    string_template = StringTemplate(prompt_context=context, template=TEMPLATE)
    final_string_prompt = string_template.render()
    logger.info(final_string_prompt)
