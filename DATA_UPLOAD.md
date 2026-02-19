# Data Upload Requirements for CLBench

## Overview
This environment requires the CL-Bench dataset to be uploaded to OpenReward cloud storage at `/orwd_data/clbench/`.

## Directory Structure
```
/orwd_data/clbench/
└── clbench_tasks.parquet
```

## File Description

### `clbench_tasks.parquet`
- **Content**: Complete CL-Bench dataset with 1,899 evaluation tasks
- **Size**: ~5-10 MB (estimated)
- **Format**: Parquet (pandas-compatible)
- **Source**: HuggingFace `tencent/CL-bench` (split="train")
- **Fields**:
  - `messages`: List of conversation messages (system, user, assistant)
  - `rubrics`: List of evaluation criteria (3-114 per task)
  - `metadata`: Task identifiers and categories

## How to Generate This File

Run the data preparation script locally:

```bash
# Install dependencies
pip install datasets pandas pyarrow

# Download and convert dataset
python download_data.py
```

This will:
1. Download the dataset from HuggingFace (split="train", which contains evaluation tasks)
2. Convert to parquet format for efficient loading
3. Save as `clbench_tasks.parquet` in the project directory
4. Display file size and validation info

## Upload Instructions

### Step 1: Create Namespace
1. Go to https://openreward.ai
2. Create a new namespace: `EnvCommons/clbench`
3. Configure namespace settings

### Step 2: Upload Data File
1. Navigate to namespace storage settings
2. Upload `clbench_tasks.parquet` to `/orwd_data/clbench/`
3. Verify the file path is exactly: `/orwd_data/clbench/clbench_tasks.parquet`

### Step 3: Verify Upload
After uploading, the environment server logs should display:
```
[CLBench] Loaded 1899 tasks from /orwd_data/clbench/clbench_tasks.parquet
[CLBench] Prepared 1899 task specifications
```

## Troubleshooting

### File Not Found Error
If you see: `FileNotFoundError: /orwd_data/clbench/clbench_tasks.parquet`

**Solutions**:
- Verify file is at the exact path: `/orwd_data/clbench/clbench_tasks.parquet`
- Check namespace has read permissions for the storage
- Ensure file upload completed successfully (check file size matches local file)

### Wrong Number of Tasks
If logs show a different number of tasks than 1,899:

**Solutions**:
- Re-download the dataset using `python download_data.py`
- Verify you're using the latest version of the `tencent/CL-bench` dataset
- Check the parquet file integrity

### Permission Errors
If the environment cannot read the file:

**Solutions**:
- Verify namespace storage permissions are configured correctly
- Ensure the environment has access to the `/orwd_data` mount
- Check bucket configuration in deployment settings

## Local Development
For local testing, you don't need to upload to cloud storage. The environment will automatically fall back to reading from the project directory:
```
/home/ross/Documents/or_envs/newenvs/clbench/clbench_tasks.parquet
```

This allows you to test locally before deploying to production.
