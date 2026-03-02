# CL-Bench

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/CL-Bench) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/tencent/CL-bench)

## Description

CL-Bench (Context Learning Benchmark) is an environment for evaluating the ability of language models to learn from novel context provided at inference time and apply that knowledge to answer questions. The benchmark contains 500 complex contexts, 1,899 tasks, and 31,607 verification rubrics crafted by domain experts. Tasks span four categories: domain knowledge reasoning, novel rule systems, procedural task following, and empirical discovery.

## Capabilities

- Context learning from novel information presented at inference time
- Applying newly acquired knowledge to answer domain-specific questions
- Multi-rubric evaluation against expert-crafted verification criteria
- Domain-specific reasoning across four distinct task categories

## Compute Requirements

This is a single-turn environment with no sandbox.

## License

[CL-Bench License](https://huggingface.co/datasets/tencent/CL-bench/blob/main/LICENSE.txt) (research/benchmarking only, no training).

## Tasks

There are 1,899 tasks in a single "test" split, drawn from 500 unique contexts sourced from the [tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench) dataset. Each task provides a system context containing novel information, rules, or domain knowledge, followed by a question that the agent must answer using the provided context. Tasks are evaluated against 3 to 114 rubrics per task (averaging approximately 16.6), for a total of 31,607 rubrics across the benchmark.

The four task categories are:

1. **Domain Knowledge Reasoning** -- Applying subject-specific expertise presented in the context.
2. **Novel Rule Systems** -- Following explicitly defined rules provided in the context.
3. **Procedural Task Following** -- Executing multi-step procedures described in the context.
4. **Empirical Discovery** -- Pattern recognition and simulation based on contextual data.

## Reward Structure

CL-Bench uses binary reward scoring (1.0 or 0.0). Each task has 3 to 114 verification rubrics (averaging ~16.6 rubrics per task) crafted by domain experts. Rubrics specify precise criteria the answer must satisfy, such as "mentions X concept", "correctly applies rule Y", or "includes numerical result Z". All rubrics for a task must pass for the agent to receive a reward of 1.0. If any single rubric fails, the reward is 0.0. Each rubric is graded independently by gpt-5-mini, and detailed per-rubric pass/fail feedback is returned.

## Data

Task data is stored in `clbench_tasks.parquet`, sourced from the HuggingFace dataset [tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench). In production, data is stored on the OpenReward platform at `/orwd_data/clbench/`.

## Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `submit_answer` | `answer: str` | Submit your final answer for evaluation against all rubrics. Ends the episode. |

## Time Horizon

CL-Bench is a single-turn environment. The agent reads the context and question, then submits one answer via the `submit_answer` tool.

## Environment Difficulty

The original paper evaluates ten frontier language models on CL-Bench:

| Model | Success Rate |
|-------|--------------|
| GPT-5.1 | 23.7% |
| Average (10 models) | 17.2% |

The strict all-rubrics-must-pass criterion and complexity of context learning tasks make this a challenging benchmark.

## Other Environment Requirements

CL-Bench requires an `openai_api_key` secret for gpt-5-mini rubric evaluation. Pass this via the session secrets:

```python
async with environment.session(
    task=task,
    secrets={"openai_api_key": "sk-..."}
) as session:
    ...
```

## Safety

CL-Bench does not present safety concerns. The agent processes text contexts and submits text answers. There is no file system access, no network interaction, and no sandbox execution.

## Citations

```bibtex
@article{an2025clbench,
  title={CL-bench: A Benchmark for Context Learning},
  author={An, Deliang and Wang, Ningxuan and Yang, Xiaotong and Dong, Liang and Liu, Yuhong},
  journal={arXiv preprint arXiv:2602.03587},
  year={2025}
}
```
