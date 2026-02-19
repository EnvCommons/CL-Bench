"""
Test grader directly without running agent.

This script tests the submit_answer tool by directly calling it with a sample
submission, allowing you to verify the grader evaluation works correctly.

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_grader.py

Optional:
    export BASE_URL="http://localhost:8080"  # For local testing
    export TASK_INDEX="0"  # Which task to test (default: 0)
"""

import asyncio
import json
import os

from openreward import AsyncOpenReward


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
TASK_INDEX = int(os.environ.get("TASK_INDEX", "0"))


async def main() -> None:
    """Test submit_answer with sample submission."""

    or_client = AsyncOpenReward()

    environment = or_client.environments.get(name="local/CLBench", base_url=BASE_URL)

    tasks = await environment.list_tasks(split="test")
    print(f"Found {len(tasks)} tasks")

    task = tasks[TASK_INDEX]
    print(f"Testing task index: {TASK_INDEX}")
    print(f"Task ID: {task['task_id']}")
    print(f"Category: {task['context_category']}")
    print()

    async with environment.session(
        task=task, secrets={"openai_api_key": OPENAI_API_KEY}
    ) as session:
        # Get prompt
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text if isinstance(prompt, list) else prompt

        print("=" * 60)
        print("PROMPT (first 500 chars)")
        print("=" * 60)
        print(prompt_text[:500] + ("..." if len(prompt_text) > 500 else ""))
        print()

        # Submit a test answer
        test_answer = input(
            "Enter your test answer (or press Enter for default test): "
        ).strip()
        if not test_answer:
            test_answer = (
                "This is a test submission for grader evaluation. "
                "The grader should evaluate this against the rubrics."
            )

        print(f"\nSubmitting answer: {test_answer[:100]}{'...' if len(test_answer) > 100 else ''}")
        print()

        tool_result = await session.call_tool("submit_answer", {"answer": test_answer})

        print("=" * 60)
        print("GRADING RESULT")
        print("=" * 60)
        if tool_result.blocks:
            print(tool_result.blocks[0].text)

        print()
        print(f"Reward: {tool_result.reward}")
        print(f"Finished: {tool_result.finished}")

        if tool_result.metadata:
            print()
            print("=" * 60)
            print("METADATA (summary)")
            print("=" * 60)
            print(f"Task ID: {tool_result.metadata.get('task_id')}")
            print(
                f"Rubrics passed: {tool_result.metadata.get('rubrics_passed')}/{tool_result.metadata.get('rubrics_total')}"
            )
            print(f"All passed: {tool_result.metadata.get('all_passed')}")

            # Show first 3 rubric results as examples
            rubric_results = tool_result.metadata.get("rubric_results", [])
            if rubric_results:
                print()
                print("Sample rubric results (first 3):")
                for r in rubric_results[:3]:
                    status = "✅" if r.get("passed") else "❌"
                    print(
                        f"  {status} Rubric {r.get('rubric_index')}: {r.get('reasoning', 'N/A')[:100]}"
                    )


if __name__ == "__main__":
    asyncio.run(main())
