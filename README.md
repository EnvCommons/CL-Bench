# CLBench - Context Learning Benchmark

OpenReward environment for evaluating agent ability to learn from novel context and apply knowledge to questions.

## Overview

CL-Bench (Context Learning Benchmark) tests whether language models can acquire and apply novel information presented within the prompt, rather than relying on pre-trained knowledge.

- **Dataset**: 1,899 tasks from 500 unique contexts
- **Source**: HuggingFace [`tencent/CL-bench`](https://huggingface.co/datasets/tencent/CL-bench)
- **Evaluation**: Binary scoring (1.0 if ALL rubrics pass, else 0.0)
- **Grader**: GPT-5-mini with structured rubric evaluation
- **Difficulty**: Challenging benchmark (frontier models achieve 17-24% success rate)

## Task Categories

Tasks evaluate four primary dimensions:

1. **Domain Knowledge Reasoning** - Applying subject-specific expertise
2. **Novel Rule Systems** - Following explicitly defined rules
3. **Procedural Task Following** - Executing multi-step procedures
4. **Empirical Discovery** - Pattern recognition and simulation

## Task Format

Each task provides:
1. **System context**: Novel information, rules, or domain knowledge
2. **User question**: Task to complete using the context
3. **Evaluation rubrics**: 3-114 criteria (hidden from agent, used for grading)

## Scoring System

- **Reward = 1.0**: ALL rubrics pass ✅
- **Reward = 0.0**: ANY rubric fails ❌
- **Grading**: LM-as-judge using GPT-5-mini
- **Feedback**: Detailed per-rubric pass/fail with reasoning

This strict evaluation ensures thorough understanding and application of context.

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_data.py
```

This will:
- Download 1,899 tasks from HuggingFace
- Convert to parquet format
- Save as `clbench_tasks.parquet`
- Display file size and next steps

### 3. Run Server

```bash
export OPENAI_API_KEY="sk-..."  # Required for grader
python server.py
```

Server will start on `http://0.0.0.0:8000`

### 4. Test with Agent

```bash
# Test with OpenAI agent
export OPENAI_API_KEY="sk-..."
python test_agent.py

# Optional: specify model and task
export MODEL_NAME="gpt-5.2"
export TASK_INDEX="5"
python test_agent.py
```

### 5. Test Grader Directly

```bash
export OPENAI_API_KEY="sk-..."
python test_grader.py
```

This allows you to test the grader evaluation without running a full agent.

## Docker

### Build

```bash
docker build -t clbench:latest .
```

### Run (Local Testing)

```bash
docker run -p 8080:8000 \
  -e OPENAI_API_KEY="sk-..." \
  -v $(pwd)/clbench_tasks.parquet:/app/clbench_tasks.parquet \
  clbench:latest
```

Note: Mount local data file for testing. In production, data comes from `/orwd_data/clbench/`.

## Deployment

### Prerequisites
1. Create namespace `EnvCommons/clbench` on https://openreward.ai
2. Upload data to cloud storage (see [DATA_UPLOAD.md](DATA_UPLOAD.md))

### GitHub Setup

```bash
# Source GitHub token
source /home/ross/Documents/or_envs/newenvs/.env

# Create repository
gh repo create EnvCommons/clbench --public --source=. --remote=origin

# Push code
git add -A
git commit -m "Initial CLBench environment implementation"
git push -u origin main
```

### OpenReward Configuration
1. Go to https://openreward.ai/environments/new
2. Connect GitHub repository: `EnvCommons/clbench`
3. Set namespace: `EnvCommons/clbench`
4. Configure deployment settings
5. Verify data is uploaded to `/orwd_data/clbench/`

### Verify Deployment
Check logs for:
```
[CLBench] Loaded 1899 tasks from /orwd_data/clbench/clbench_tasks.parquet
[CLBench] Prepared 1899 task specifications
```

## Environment Details

### Namespace
- **Production**: `EnvCommons/clbench`
- **Local**: `local/CLBench`

### Splits
- **test**: All 1,899 evaluation tasks

### Tools

#### `submit_answer(answer: str)`
Submit final answer for evaluation.

**Parameters**:
- `answer` (string): Your complete answer to the question

**Returns**:
- Display text with overall result and per-rubric feedback
- Reward: 1.0 if all rubrics pass, 0.0 otherwise
- Finished: `true` (single submission per task)

**Metadata**:
- `task_id`: Task identifier
- `rubrics_passed`: Number of rubrics that passed
- `rubrics_total`: Total number of rubrics
- `all_passed`: Boolean indicating if ALL rubrics passed
- `rubric_results`: Detailed per-rubric evaluation
- `reference_answer`: Reference solution (for analysis)

### Secrets Required
- `openai_api_key`: Required for GPT-5-mini grader

Pass via session:
```python
async with environment.session(task=task, secrets={"openai_api_key": "sk-..."}) as session:
    ...
```

## File Structure

```
clbench/
├── server.py              # Main environment + server
├── download_data.py       # Data preparation script
├── test_agent.py          # OpenAI test client
├── test_grader.py         # Direct grader testing
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── DATA_UPLOAD.md        # Upload instructions
├── README.md             # This file
└── .gitignore           # Exclude data files
```

## Example Usage

### Python Client

```python
import asyncio
from openai import AsyncOpenAI
from openreward import AsyncOpenReward

async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key="sk-...")

    # Connect to environment
    environment = or_client.environments.get(
        name="EnvCommons/clbench"
    )

    # Get task
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    # Run session
    async with environment.session(
        task=tasks[0],
        secrets={"openai_api_key": "sk-..."}
    ) as session:
        prompt = await session.get_prompt()

        # ... your agent logic ...

        result = await session.call_tool(
            "submit_answer",
            {"answer": "My answer based on the context"}
        )

        print(f"Reward: {result.reward}")

asyncio.run(main())
```

## Dataset Information

### Source
- **HuggingFace**: `tencent/CL-bench`
- **Split**: "train" (contains evaluation tasks despite name)
- **Paper**: [CL-Bench: Evaluating Context Learning](https://arxiv.org/abs/...)

### Statistics
- **Total tasks**: 1,899
- **Contexts**: 500 unique contexts
- **Rubrics per task**: 3-114 (average ~16)
- **Total rubrics**: 31,607

### License
See dataset page on HuggingFace for license information.

## Troubleshooting

### FileNotFoundError
**Error**: `FileNotFoundError: clbench_tasks.parquet`

**Solution**: Run `python download_data.py` to download the dataset.

### Grading Failures
**Error**: "Grading failed due to error"

**Solutions**:
- Verify `OPENAI_API_KEY` is set and valid
- Check API rate limits
- Ensure internet connectivity (unless using local proxy)

### Wrong Reward
**Issue**: Expecting 1.0 but got 0.0

**Explanation**: ALL rubrics must pass for reward=1.0. Check the detailed feedback to see which rubrics failed.

## Performance Notes

- **Grading time**: 5-15 seconds per submission (depends on rubric count)
- **API costs**: ~$0.01-0.05 per task (GPT-5-mini pricing)
- **Difficulty**: High - frontier models achieve 17-24% success rate

## Contributing

For issues or improvements, please open an issue on the GitHub repository.

## References

- [OpenReward Platform](https://openreward.ai)
- [CL-Bench Dataset](https://huggingface.co/datasets/tencent/CL-bench)
- [OpenReward Documentation](https://docs.openreward.ai)
