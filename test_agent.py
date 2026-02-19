"""
Test agent for CLBench environment using OpenAI Responses API.

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_agent.py

Optional:
    export MODEL_NAME="gpt-5.2"  # Default: gpt-5
    export BASE_URL="http://localhost:8080"  # For local testing
    export TASK_INDEX="0"  # Which task to test (default: 0)
"""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
TASK_INDEX = int(os.environ.get("TASK_INDEX", "0"))


async def main() -> None:
    """Run agent on CLBench task."""

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to environment
    environment = or_client.environments.get(name="local/CLBench", base_url=BASE_URL)

    # Get tasks and tools
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")
    print(f"Testing task index: {TASK_INDEX}")
    print()

    # Run session
    async with environment.session(
        task=tasks[TASK_INDEX], secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        # Get prompt
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text if isinstance(prompt, list) else prompt

        print("=" * 60)
        print("PROMPT (first 1000 chars)")
        print("=" * 60)
        print(prompt_text[:1000] + ("..." if len(prompt_text) > 1000 else ""))
        print()

        # Initialize conversation
        input_list = [{"role": "user", "content": prompt_text}]
        finished = False
        turn = 0

        while not finished and turn < 10:  # Max 10 turns safety limit
            turn += 1
            print(f"--- Turn {turn} ---")

            # Get model response
            response = await oai_client.responses.create(
                model=MODEL_NAME, tools=tools, input=input_list
            )

            print(f"Model: {MODEL_NAME}")
            for item in response.output:
                if item.type == "text":
                    print(f"Text: {item.text[:200]}{'...' if len(item.text) > 200 else ''}")
                elif item.type == "function_call":
                    print(f"Tool Call: {item.name}")
                    args = json.loads(str(item.arguments))
                    answer_preview = args.get("answer", "")[:100]
                    print(f"Answer (preview): {answer_preview}{'...' if len(answer_preview) == 100 else ''}")
            print()

            # Add model output to conversation
            input_list += response.output

            # Process tool calls
            for item in response.output:
                if item.type == "function_call":
                    # Execute tool
                    tool_result = await session.call_tool(
                        item.name, json.loads(str(item.arguments))
                    )

                    finished = tool_result.finished

                    # Add tool result to conversation
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(
                                {
                                    "result": (
                                        tool_result.blocks[0].text
                                        if tool_result.blocks
                                        else ""
                                    )
                                }
                            ),
                        }
                    )

                    print("=" * 60)
                    print("TOOL RESULT")
                    print("=" * 60)
                    if tool_result.blocks:
                        print(tool_result.blocks[0].text)
                    print()
                    print(f"Reward: {tool_result.reward}")
                    print(f"Finished: {tool_result.finished}")
                    print()

                    if finished:
                        break

            # Safety: break if no tool calls and no continuation
            if not any(item.type == "function_call" for item in response.output):
                print("No tool calls made, ending session")
                break

    print("=" * 60)
    print("Session completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
