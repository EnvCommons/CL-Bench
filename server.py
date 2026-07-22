"""
CL-Bench OpenReward Environment

Context Learning Benchmark evaluating agent ability to learn from novel context
and apply knowledge to questions. Uses LM-as-judge with multi-rubric evaluation.

Dataset: 1,899 tasks from 500 contexts
Scoring: Binary (1.0 if ALL rubrics pass, 0.0 otherwise)
Grader: GPT-5-mini
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import openai
import pandas as pd
from pydantic import BaseModel, Field

from openreward.environments import (
    Environment,
    JSONObject,
    Server,
    TextBlock,
    ToolOutput,
    terminal,
    tool,
)


# --- Pydantic Models ---


class TaskSpec(BaseModel):
    """Task specification (public)"""

    task_id: str


class SubmitAnswerInput(BaseModel):
    """Input for submit_answer tool"""

    answer: str = Field(..., description="Your final answer to the question")


# --- Data Loading (Module Level) ---

# Path checking: production vs development
if os.path.exists("/orwd_data"):
    DATA_PATH = Path("/orwd_data")
else:
    DATA_PATH = Path(__file__).parent

# Load dataset
TASKS_DF = pd.read_parquet(DATA_PATH / "clbench_tasks.parquet")

# Handle NaN values and clean up (important for JSON serialization)
if '__index_level_0__' in TASKS_DF.columns:
    TASKS_DF = TASKS_DF.drop('__index_level_0__', axis=1)

TASKS_DF = TASKS_DF.fillna('')

print(f"[CLBench] Loaded {len(TASKS_DF)} tasks from {DATA_PATH / 'clbench_tasks.parquet'}")

# Create PUBLIC task specs (no answers/rubrics)
TASK_SPECS = []
for idx, row in TASKS_DF.iterrows():
    task_id = row["metadata"]["task_id"]
    TASK_SPECS.append({
        "task_id": task_id,
        "context_category": row["metadata"].get("context_category", "Unknown"),
        "sub_category": row["metadata"].get("sub_category", "Unknown"),
    })

# Create PRIVATE storage (backend only - includes answers/rubrics)
TASK_DATA = {}
for idx, row in TASKS_DF.iterrows():
    task_id = row["metadata"]["task_id"]
    TASK_DATA[task_id] = {
        "messages": row["messages"],  # list of {role, content}
        "rubrics": row["rubrics"],  # list of rubric strings
        "metadata": row["metadata"],
    }

print(f"[CLBench] Prepared {len(TASK_SPECS)} task specifications")


# --- Environment Class ---


class CLBench(Environment):
    """
    CL-Bench: Context Learning Benchmark

    1,899 tasks evaluating agent ability to learn from novel context and apply
    knowledge to questions. Tasks span domain knowledge, rule systems, procedural
    tasks, and empirical discovery.

    Evaluation:
    - Binary scoring: 1.0 if ALL rubrics pass, 0.0 otherwise
    - LM-as-judge using gpt-5-mini
    - Rubric count varies: 3-114 rubrics per task
    """

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available splits"""
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task specifications for split"""
        if split != "test":
            return []
        return TASK_SPECS

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """Initialize environment for a specific task"""
        super().__init__(task_spec)

        # Validate task spec
        self.validated = TaskSpec.model_validate(task_spec)

        # Get task data from private storage
        if self.validated.task_id not in TASK_DATA:
            raise ValueError(f"Task not found: {self.validated.task_id}")

        self.task_data = TASK_DATA[self.validated.task_id]

        # Initialize OpenAI client for grading
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required in secrets (key: 'openai_api_key'). "
                "This is used for LM-as-judge evaluation with GPT-5-mini."
            )

        self.grader_client = openai.AsyncClient(api_key=api_key)

        # Track submission state
        self.submitted = False

    async def get_prompt(self) -> list[TextBlock]:
        """
        Build prompt from messages (system + user).

        CL-Bench format:
        - messages[0]: {"role": "system", "content": "Context..."}
        - messages[1]: {"role": "user", "content": "Question..."}
        - messages[2]: {"role": "assistant", "content": "Reference (hidden)"}
        """
        messages = self.task_data["messages"]

        # Combine system and user messages
        prompt_parts = []

        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"CONTEXT:\n{msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"QUESTION:\n{msg['content']}")
            # Skip assistant message (reference solution - hidden from agent)

        # Add submission instruction
        prompt_parts.append(
            "\nINSTRUCTIONS:\n"
            "Read the context carefully and answer the question based on the "
            "information provided. Reply with your final answer as an ordinary "
            "message (no tool call)."
        )

        full_prompt = "\n\n".join(prompt_parts)

        return [TextBlock(type="text", text=full_prompt)]

    @terminal
    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit final answer for evaluation against rubrics.

        Evaluation:
        - All rubrics must pass for reward=1.0
        - Any rubric failure results in reward=0.0
        - Provides detailed per-rubric feedback
        """
        if self.submitted:
            return ToolOutput(
                blocks=[
                    TextBlock(
                        type="text",
                        text="You have already submitted an answer for this task.",
                    )
                ],
                metadata={"error": "already_submitted"},
                reward=0.0,
                finished=True,
            )

        self.submitted = True

        # Grade submission
        grading_result = await self._grade_submission(params.answer)

        return ToolOutput(
            blocks=[TextBlock(type="text", text=grading_result["display_text"])],
            metadata=grading_result["metadata"],
            reward=grading_result["reward"],
            finished=True,
        )

    async def _grade_submission(self, submission: str) -> dict[str, Any]:
        """
        Grade submission using gpt-5-mini against all rubrics.

        Returns:
            dict with keys: display_text, metadata, reward
        """
        messages = self.task_data["messages"]
        rubrics = self.task_data["rubrics"]

        # Extract reference answer
        reference_answer = next(
            (msg["content"] for msg in messages if msg["role"] == "assistant"),
            "No reference provided",
        )

        # Extract context and question
        system_context = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), ""
        )
        user_question = next(
            (msg["content"] for msg in messages if msg["role"] == "user"), ""
        )

        # Grade all rubrics in parallel
        try:
            rubric_results = await self._grade_all_rubrics(
                context=system_context,
                question=user_question,
                reference_answer=reference_answer,
                submission=submission,
                rubrics=rubrics,
            )
        except Exception as e:
            # Fallback: fail with error message
            return {
                "display_text": f"⚠️ Grading failed due to error: {str(e)}\n\nDefaulting to FAIL.",
                "metadata": {
                    "task_id": self.validated.task_id,
                    "error": str(e),
                    "rubrics_passed": 0,
                    "rubrics_total": len(rubrics),
                },
                "reward": 0.0,
            }

        # Determine overall pass/fail (ALL rubrics must pass)
        all_passed = all(r["passed"] for r in rubric_results)
        reward = 1.0 if all_passed else 0.0

        # Build display text
        display_text = self._build_display_text(rubric_results, all_passed)

        return {
            "display_text": display_text,
            "metadata": {
                "task_id": self.validated.task_id,
                "rubrics_passed": sum(1 for r in rubric_results if r["passed"]),
                "rubrics_total": len(rubrics),
                "all_passed": all_passed,
                "rubric_results": rubric_results,
                "reference_answer": reference_answer,
            },
            "reward": reward,
        }

    async def _grade_all_rubrics(
        self,
        context: str,
        question: str,
        reference_answer: str,
        submission: str,
        rubrics: list[str],
    ) -> list[dict[str, Any]]:
        """
        Grade all rubrics in parallel using asyncio.gather.

        Returns:
            list of {rubric_index, passed, reasoning}
        """
        # Truncate long context (keep first 2000 chars to avoid token limits)
        context_display = context[:2000]
        if len(context) > 2000:
            context_display += "...[truncated]"

        # Create grading tasks for each rubric
        grading_tasks = [
            self._grade_single_rubric(
                context=context_display,
                question=question,
                reference_answer=reference_answer,
                submission=submission,
                rubric=rubric,
                rubric_index=i + 1,
            )
            for i, rubric in enumerate(rubrics)
        ]

        # Execute all grading tasks in parallel
        results = await asyncio.gather(*grading_tasks, return_exceptions=True)

        # Process results and handle any exceptions
        rubric_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # If a rubric grading failed, mark as failed
                rubric_results.append({
                    "rubric_index": i + 1,
                    "passed": False,
                    "reasoning": f"Grading error: {str(result)}",
                })
            else:
                rubric_results.append(result)

        return rubric_results

    async def _grade_single_rubric(
        self,
        context: str,
        question: str,
        reference_answer: str,
        submission: str,
        rubric: str,
        rubric_index: int,
    ) -> dict[str, Any]:
        """
        Grade a single rubric using gpt-5-mini.

        Returns:
            dict with {rubric_index, passed, reasoning}
        """
        prompt = f"""You are evaluating whether a submission meets a specific rubric criterion.

CONTEXT (provided to agent):
{context}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

SUBMISSION TO EVALUATE:
{submission}

RUBRIC TO EVALUATE:
{rubric}

INSTRUCTIONS:
1. Determine if the submission meets this specific rubric criterion
2. Compare against the reference answer where helpful
3. Be fair but strict - the rubric must be fully satisfied
4. Consider that the submission may be correct even if it differs from the reference
5. Return your evaluation in JSON format

OUTPUT FORMAT (respond ONLY with valid JSON):
{{
  "passed": true or false,
  "reasoning": "Brief explanation of why this rubric passed or failed"
}}

Begin evaluation:"""

        max_retries = 2

        for attempt in range(max_retries):
            try:
                response = await self.grader_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},  # Force JSON
                )

                grader_output = response.choices[0].message.content or "{}"

                # Parse JSON
                result = json.loads(grader_output)

                # Validate structure
                if "passed" not in result:
                    raise ValueError("Missing 'passed' in grader output")

                # Add rubric index to result
                return {
                    "rubric_index": rubric_index,
                    "passed": result["passed"],
                    "reasoning": result.get("reasoning", "No reasoning provided"),
                }

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"[GRADER] Rubric {rubric_index}: JSON parse error, retrying... ({e})")
                    await asyncio.sleep(1)
                    continue
                else:
                    # Fallback: mark as failed
                    print(f"[GRADER] Rubric {rubric_index}: JSON parse failed after retries")
                    return {
                        "rubric_index": rubric_index,
                        "passed": False,
                        "reasoning": f"Grader returned invalid JSON: {e}",
                    }

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[GRADER] Rubric {rubric_index}: Error, retrying... ({e})")
                    await asyncio.sleep(1)
                    continue
                else:
                    # Re-raise to be caught by gather
                    raise

    def _build_display_text(self, rubric_results: list[dict[str, Any]], all_passed: bool) -> str:
        """Build user-facing feedback text"""
        passed_count = sum(1 for r in rubric_results if r["passed"])
        total_count = len(rubric_results)

        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            "",
            f"Overall: {'✅ PASS' if all_passed else '❌ FAIL'}",
            f"Rubrics passed: {passed_count}/{total_count}",
            "",
            "Note: ALL rubrics must pass to receive credit (reward = 1.0)",
            "",
            "=" * 60,
            "RUBRIC DETAILS",
            "=" * 60,
            "",
        ]

        # Show each rubric result
        for i, r in enumerate(rubric_results, 1):
            status = "✅" if r["passed"] else "❌"
            lines.append(f"{status} Rubric {i}: {'PASS' if r['passed'] else 'FAIL'}")

            # Show reasoning (truncate if too long)
            reasoning = r.get("reasoning", "No reasoning provided")
            if len(reasoning) > 150:
                reasoning = reasoning[:150] + "..."
            lines.append(f"   └─ {reasoning}")
            lines.append("")

        return "\n".join(lines)


# --- Server Startup ---

if __name__ == "__main__":
    server = Server([CLBench])
    server.run()
