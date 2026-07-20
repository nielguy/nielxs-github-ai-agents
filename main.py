import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import  MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import ToolException

load_dotenv()

MCP_SERVERS_CONFIG = {
    "github-reader": {
        "url": "http://127.0.0.1:8000/sse",
        "transport": "sse"
    }
}

def apply_input_guardrail(inputs: dict, **kwargs) -> dict:
    user_text = inputs["messages"][-1].content

    if "secret_token" in user_text:
        raise ValueError("Guardrail Blocked: Input contains sensitive info.")

    return inputs

def apply_output_guardrail(message) -> bool:
    """Checks tool calls before they run. Returns True if safe."""
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            arguments = tool_call.get("args", {})
            repo = arguments.get("repo", "") or arguments.get("repository", "")

            if repo and "anthropics/" not in repo and "langchain" not in repo:
                print(f"Guardrail Blocked Tool Call: Unauthorized repo '{repo}'")
                return False
    return True

def wrap_tool_with_guard(tool):
    """Wraps an existing LangChain tool with input argument validation."""
    original_coroutine = tool.coroutine

    async def guarded_coroutine(*args, **kwargs):

        repo = kwargs.get("repo", "") or kwargs.get("repository", "")
        if repo and "anthropics/" not in repo and "langchain" not in repo:
            raise ToolException(f"Guardrail Blocked: Unauthorized repository access to '{repo}'.")

        return await original_coroutine(*args, **kwargs)

    tool.coroutine = guarded_coroutine
    tool.handle_tool_error = True
    return tool


async def main():
    print("Connecting to MCP servers via SSE transport...")
    mcp_client = MultiServerMCPClient(MCP_SERVERS_CONFIG)

    try:
        print(f"Fetching tools from all endpoints...")
        langchain_tools = await mcp_client.get_tools()
        guarded_tools = [wrap_tool_with_guard(t) for t in langchain_tools]
        print(f"Successfully loaded {len(guarded_tools)} tools.")

        for tool in langchain_tools:
            print(f" - Found tool: {tool.name}")

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2
        )

        agent = create_agent(llm, tools=langchain_tools)

        # user_prompt = (
        #     "What tools do you have available?"
        # )

        user_prompt = (
            "Read the README from the anthropics/anthropic-sdk-python repository and give me a one-paragraph summary."
        )

        print(f"\n Invoking Agent with prompt: '{user_prompt}'\n")

        raw_inputs = {"messages": [HumanMessage(content=user_prompt)]}

        input_guardrail_runnable = RunnableLambda(apply_input_guardrail)

        print("Running input guardrail pipeline...")

        inputs = await input_guardrail_runnable.ainvoke(raw_inputs)

        async for chunk in agent.astream(inputs, stream_mode="values"):
            if "messages" in chunk:
                last_message = chunk["messages"][-1]

                if last_message.type == "ai":
                    if not apply_output_guardrail(last_message):
                        print(f"Execution halted: Output/Tool call violated guardrail policies.")
                        break

                last_message.pretty_print()

    finally:
        print("Closing MCP servers...")



if __name__ == "__main__":
    asyncio.run(main())




# uv run main.py
# Connecting to MCP servers via SSE transport...
# Fetching tools from all endpoints...
# Successfully loaded 1 tools.
#  - Found tool: read_github_file
# /Users/nielxs/Downloads/github/nielxs-github-ai-agents/main.py:43: LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent`. Deprecated in LangGraph V1.0 to be removed in V2.0.
#   agent = create_react_agent(llm, tools=langchain_tools)
#
#  Invoking Agent with prompt: 'What tools do you have available?'
#
# ================================ Human Message =================================
#
# What tools do you have available?
# ================================== Ai Message ==================================
#
# I have access to the following tools:
#
# 1. **functions.read_github_file**: This tool allows me to read the contents of a file from a GitHub repository. It requires the repository name, file path, and optionally a branch or commit reference.
#
# 2. **multi_tool_use.parallel**: This tool allows me to execute multiple tools simultaneously, provided they can operate in parallel. It requires specifying the tools to be used and their respective parameters.
# Closing MCP servers...




#  uv run main.py
# Connecting to MCP servers via SSE transport...
# Fetching tools from all endpoints...
# Successfully loaded 1 tools.
#  - Found tool: read_github_file
#
#  Invoking Agent with prompt: 'Read the README from the anthropics/anthropic-sdk-python repository and give me a one-paragraph summary.'
#
# ================================ Human Message =================================
#
# Read the README from the anthropics/anthropic-sdk-python repository and give me a one-paragraph summary.
# ================================== Ai Message ==================================
# Tool Calls:
#   read_github_file (call_s0d)
#  Call ID: call_s0d
#   Args:
#     repo: anthropics/anthropic-sdk-python
#     path: README.md
# ================================= Tool Message =================================
# Name: read_github_file
#
# [{'type': 'text', 'text': 'File: anthropics/anthropic-sdk-python/README.md (ref: main)\n\n# Claude SDK for Python\n\n[![PyPI version](https://img.shields.io/pypi/v/anthropic.svg)](https://pypi.org/project/anthropic/)\n\nThe Claude SDK for Python provides access to the [Claude API](https://docs.anthropic.com/en/api/) from Python applications.\n\n## Documentation\n\nFull documentation is available at **[platform.claude.com/docs/en/api/sdks/python](https://platform.claude.com/docs/en/api/sdks/python)**.\n\n## Installation\n\n```sh\npip install anthropic\n```\n\n## Getting started\n\n```python\nimport os\nfrom anthropic import Anthropic\n\nclient = Anthropic(\n    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted\n)\n\nmessage = client.messages.create(\n    max_tokens=1024,\n    messages=[\n        {\n            "role": "user",\n            "content": "Hello, Claude",\n        }\n    ],\n\n    model="claude-opus-4-6",\n)\n\nprint(message.content)\n```\n\n\n## Requirements\n\nPython 3.9+\n\n## Contributing\n\nSee [CONTRIBUTING.md](./CONTRIBUTING.md).\n\n## License\n\nThis project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.\n', 'id': 'lc_58a'}]
# ================================== Ai Message ==================================
#
# The Claude SDK for Python allows developers to access the Claude API from Python applications. It provides a straightforward installation process via pip and requires Python 3.9 or higher. The SDK enables users to interact with the API by creating messages and specifying parameters such as `max_tokens` and the model type. Full documentation is available online, and the project is open-source, licensed under the MIT License.
# Closing MCP servers...
