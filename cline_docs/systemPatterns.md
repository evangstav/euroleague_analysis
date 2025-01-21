# System Patterns

## Architecture Overview
The project follows a modular architecture with clear separation of concerns:

### Data Collection
- `euroleague_data_collector.py`: Handles data collection from Euroleague sources
- Uses SQL queries for data transformation and feature extraction

### Feature Engineering
- Located in `feature_engineering/` module
- Follows a builder pattern for feature construction
- Uses views pattern for organizing different types of statistics
- Components:
  - `builder.py`: Orchestrates feature creation
  - `database.py`: Manages database operations
  - `views/`: Contains different statistical views
    - `base.py`: Base view implementation
    - `stats.py`: Statistical calculations

### Model Training and Prediction
- `train_model.py`: Handles model training pipeline
- `predict_next_pir.py`: Implements prediction logic
- `model_config.py`: Contains model configuration

### Data Version Control
- Uses DVC for managing data and model versions
- Configuration in `dvc.yaml` and `params.yaml`

## Key Technical Decisions
1. SQL-based feature engineering for efficient data transformation
2. Modular view system for organizing different types of statistics
3. DVC integration for reproducible ML pipelines
4. Builder pattern for flexible feature construction

## Design Patterns
1. Builder Pattern: Used in feature engineering for constructing complex feature sets
2. View Pattern: Organizes different statistical calculations
3. Factory Pattern: Creates different types of features and models
4. Pipeline Pattern: Manages the flow from data collection to prediction