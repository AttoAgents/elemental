"""
Examples of using the simplified LLM Context Manager.
"""

from loguru import logger

from elemental_agents.core.context.context_manager import (
    ContextConfig,
    LLMContextManager,
    create_code_file_context,
    create_file_context,
)


# Example 1: Basic usage
def example_basic_usage() -> None:
    """
    Basic usage of the LLM Context Manager to gather context from a directory.
    """
    manager = LLMContextManager()

    # Get context data
    context_data = manager.gather_context(
        "./elemental_agents/llm", include_content=True
    )

    # Format for LLM
    formatted_context = manager.format_context(context_data)
    print("=== Basic Context ===")
    print(formatted_context)


# Example 2: Custom configuration
def example_custom_config() -> None:
    """
    Custom configuration for the LLM Context Manager to include specific file types and size limits.
    """
    config = ContextConfig(
        include_extensions=[".py", ".md"],
        max_file_size=50 * 1024,  # 50KB
        max_content_length=5000,
        include_line_numbers=True,
    )

    manager = LLMContextManager(config=config)
    context_data = manager.gather_context(
        "./elemental_agents/llm", include_content=True
    )
    formatted_context = manager.format_context(context_data)

    print("=== Custom Config Context ===")
    print(formatted_context)


# Example 3: File list only
def example_file_list_only() -> None:
    """
    Get just the file list without content.
    """
    manager = LLMContextManager()

    # Get just file list without content
    files = manager.get_file_list("./elemental_agents/llm")
    print("=== File List Only ===")
    for file_info in files:
        print(f"- {file_info.relative_path} ({file_info.size} bytes)")


# Example 4: Convenience functions
def example_convenience_functions() -> None:
    """
    Example of using convenience functions to create context from files.
    """
    # Quick context
    context = create_file_context("./elemental_agents/llm", include_content=True)
    print("=== Quick Context ===")
    print(context[:500] + "..." if len(context) > 500 else context)

    # Code-optimized context
    code_context = create_code_file_context("./elemental_agents/llm")
    print("\n=== Code Context ===")
    print(code_context[:500] + "..." if len(code_context) > 500 else code_context)


# Example 5: Integration with LLM
def example_llm_integration() -> None:
    """
    Example of integrating the context manager with an LLM for code analysis.
    """
    from elemental_agents.llm.data_model import Message
    from elemental_agents.llm.llm_factory import LLMFactory

    # Create context
    context = create_code_file_context("./elemental_agents/llm_v")

    # Create messages
    system_message = Message(
        role="system",
        content=f"""You are a code analysis assistant. Here is the current codebase:

{context}

Please analyze the code structure and provide insights.""",
    )

    user_message = Message(
        role="user", content="What is the overall architecture of this LLM framework?"
    )

    # Use with LLM (uncomment to test)
    factory = LLMFactory()
    llm = factory.create("openai|gpt-4.1")
    result = llm.run([system_message, user_message])
    logger.info("=== LLM Analysis ===")
    print(result)


if __name__ == "__main__":
    example_basic_usage()
    print("\n" + "=" * 50 + "\n")

    # example_custom_config()
    # print("\n" + "="*50 + "\n")

    # example_file_list_only()
    # print("\n" + "="*50 + "\n")

    # example_convenience_functions()
