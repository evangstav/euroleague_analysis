# Euroleague Analysis

A data analysis pipeline for Euroleague basketball using DVC and UV.

## Setup

1. Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

3. Initialize DVC:
```bash
dvc init
dvc add euroleague_data/
```

## Pipeline Stages

The analysis pipeline consists of several stages managed by DVC:

1. `extract`: Collect data from Euroleague API
2. `features`: Engineer features from raw data
3. `train`: Train prediction model
4. `analyze`: Analyze model predictions
5. `train_pgm`: Train probabilistic graphical model
6. `analyze_pgm`: Analyze PGM results

Run the full pipeline:
```bash
dvc repro
```

Run specific stages:
```bash
dvc repro <stage_name>
```

## Development

- Use UV for dependency management
- Follow DVC best practices for data versioning
- Keep params.yaml updated for configuration
