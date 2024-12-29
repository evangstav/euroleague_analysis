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

## Repository Structure

```
euroleague_analysis/
├── analyse_datasets/     # Analysis notebooks and scripts
├── models/              # Trained models and metrics
├── src/
│   └── euroleague_analysis/
│       ├── feature_engineering/  # Feature engineering pipeline
│       │   ├── views/           # SQL view definitions
│       │   ├── builder.py       # Feature builder
│       │   └── database.py      # Database operations
│       ├── sql_queries/         # SQL query templates
│       ├── train_model.py       # Model training
│       ├── predict_next_pir.py  # PIR predictions
│       └── analyze_predictions.py # Prediction analysis
├── tests/               # Test suite
├── dvc.yaml            # DVC pipeline definition
├── params.yaml         # Configuration parameters
└── requirements.txt    # Project dependencies
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

## Making Predictions

To generate PIR (Performance Index Rating) predictions for players:

```bash
python -m src.euroleague_analysis.predict_next_pir
```

This will:
1. Load the trained model
2. Generate predictions for all players
3. Display top predicted performers with details:
   - Player name and team
   - Predicted PIR and current form
   - Minutes played and points
   - Starting status and home/away context
4. Save predictions to `euroleague_data/predictions/`

## Development

- Use UV for dependency management
- Follow DVC best practices for data versioning
- Keep params.yaml updated for configuration
- Run predictions after any model changes to validate improvements

## Data Flow

1. Raw data is collected from Euroleague API
2. Features are engineered using SQL views and transformations
3. Model is trained on historical data
4. Predictions are generated for upcoming games
5. Results are analyzed for accuracy and insights

## Contributing

1. Create feature branches for new functionality
2. Update tests as needed
3. Ensure DVC pipeline stages are properly connected
4. Document any new parameters in params.yaml
5. Validate predictions before merging changes
