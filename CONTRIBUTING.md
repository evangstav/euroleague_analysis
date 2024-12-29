# Contributing Guide for Automated Agents

This guide is designed to help automated agents understand the codebase architecture and make effective changes.

## Code Architecture

### 1. Feature Engineering Pipeline
- Location: `src/euroleague_analysis/feature_engineering/`
- Pattern: SQL-first approach using DuckDB
- Key Components:
  - `views/`: SQL view definitions for feature transformations
  - `builder.py`: Orchestrates feature creation across seasons
  - `database.py`: Manages database connections and operations
  - `config.py`: Handles multi-season configuration

### 2. Model Training
- Location: `src/euroleague_analysis/train_model.py`
- Pattern: Scikit-learn pipeline with standardization
- Key Aspects:
  - Features are standardized before training
  - Model artifacts are saved using joblib
  - Metrics are tracked in models/metrics.json
  - Handles data from multiple seasons

### 3. Prediction System
- Location: `src/euroleague_analysis/predict_next_pir.py`
- Pattern: Load model → Generate features → Predict → Format output
- Key Components:
  - Player filtering (excludes team totals)
  - Feature standardization
  - Results sorting (by PIR and Round)
  - Player metadata enrichment

## Making Changes

### 1. Feature Engineering Changes
```python
# Pattern for adding new features in SQL
# Location: src/euroleague_analysis/sql_queries/
"""
WITH base_stats AS (
    SELECT *
    FROM player_stats
    WHERE Player_ID != 'Total'
),
new_feature AS (
    SELECT 
        Player_ID,
        Round,
        -- New feature calculation
    FROM base_stats
)
SELECT * FROM new_feature
"""
```

### 2. Model Changes
```python
# Pattern for modifying model pipeline
# Location: src/euroleague_analysis/train_model.py
def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ))
    ])
```

### 3. Prediction Changes
```python
# Pattern for modifying prediction output
# Location: src/euroleague_analysis/predict_next_pir.py
def format_prediction_report(predictions_df: pd.DataFrame, top_n: int = 20) -> str:
    """Format prediction results for display.
    
    Args:
        predictions_df: DataFrame with predictions
        top_n: Number of top predictions to show
        
    Returns:
        Formatted string with prediction results
    """
    report = ["=== Top Players by Predicted PIR ===\n"]
    for _, row in predictions_df.head(top_n).iterrows():
        report.append(
            f"{row['Player']} ({row['Team']})\n"
            f"  Round: {row['Round']}\n"
            # Add new fields here
        )
    return "\n".join(report)
```

## Data Flow Understanding

1. Raw Data Collection
   - Source: Euroleague API
   - Format: JSON → Parquet
   - Location: euroleague_data/raw/
   - Supports multiple seasons (e.g., 2023, 2024)

2. Feature Engineering
   - Input: Raw parquet files from multiple seasons
   - Process: Season-specific SQL transformations
   - Output: Combined feature matrices
   - Handles season boundaries in rolling calculations

3. Model Training
   - Input: Combined feature matrices from all seasons
   - Process: Scikit-learn pipeline
   - Output: Model artifacts
   - Uses full dataset across seasons

4. Prediction Generation
   - Input: Model artifacts + new data
   - Process: Feature generation → prediction
   - Output: Formatted predictions

## Common Patterns

### 1. SQL Transformations
- Use CTEs for clarity
- Filter out team totals early
- Calculate rolling statistics across seasons
- Join player metadata when needed

### 2. Feature Engineering
- Standardize numerical features
- Handle missing values explicitly
- Create rolling windows for time-series features
- Maintain feature consistency between seasons
- Handle season boundaries in rolling calculations

### 3. Model Pipeline
- Always include feature scaling
- Use consistent random seeds
- Save all model artifacts together
- Track metrics for comparison
- Train on combined multi-season dataset

### 4. Prediction Formatting
- Include player metadata
- Sort by relevant metrics
- Provide context (trends, statistics)
- Format numbers consistently

## Error Handling

1. Data Validation
```python
def validate_features(df: pd.DataFrame) -> bool:
    """Validate feature DataFrame before prediction."""
    required_columns = ['Player_ID', 'Round', 'PIR', 'minutes_played']
    return all(col in df.columns for col in required_columns)
```

2. Model Loading
```python
def load_model_artifacts(model_dir: str) -> Tuple[Pipeline, StandardScaler]:
    """Safe model artifact loading with error handling."""
    try:
        model = joblib.load(f"{model_dir}/model.joblib")
        scaler = joblib.load(f"{model_dir}/scaler.joblib")
        return model, scaler
    except FileNotFoundError as e:
        logger.error(f"Model artifacts not found: {e}")
        raise
```

## Testing Changes

1. Run the full pipeline:
```bash
dvc repro
```

2. Validate predictions:
```bash
python -m src.euroleague_analysis.predict_next_pir
```

3. Check metrics:
```bash
python -m src.euroleague_analysis.analyze_predictions
```

## Best Practices

1. Code Changes
   - Follow existing patterns
   - Maintain type hints
   - Update docstrings
   - Add logging statements

2. SQL Changes
   - Use CTEs for readability
   - Document complex calculations
   - Consider performance
   - Test with sample data
   - Handle season-specific views

3. Feature Changes
   - Document feature importance
   - Maintain scaling
   - Consider missing data
   - Test feature stability
   - Handle season transitions

4. Model Changes
   - Track metrics
   - Save artifacts
   - Document parameters
   - Test predictions
   - Validate across seasons