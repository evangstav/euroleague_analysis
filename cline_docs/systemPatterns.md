# System Patterns

## Architecture Overview
The project follows a modular architecture with clear separation of concerns:

### Data Collection
- `euroleague_data_collector.py`: Handles data collection from Euroleague sources
- Uses Parquet files for efficient data storage

### Feature Engineering
- Located in `feature_engineering/` module
- Uses transformer pattern for modular feature computation
- Leverages Polars for efficient data processing
- Components:
  - `core/`: Core abstractions and utilities
    - `base.py`: Base transformer classes
    - `registry.py`: Feature transformer registry
  - `features/`: Feature transformers
    - `basic_stats.py`: Core statistical features
    - `rolling_stats.py`: Time-series features
    - `shot_patterns.py`: Shot-related analytics
    - `playbyplay.py`: Game flow features
    - `matchups.py`: Position matchup analysis
  - `loaders/`: Data loading interfaces
    - `base.py`: Abstract loader and implementations
  - `pipeline.py`: Feature pipeline orchestration
  - `config.py`: Configuration management

### Model Training and Prediction
- `train_model.py`: Handles model training pipeline
- `predict_next_pir.py`: Implements prediction logic
- `model_config.py`: Contains model configuration

### Data Version Control
- Uses DVC for managing data and model versions
- Configuration in `dvc.yaml` and `params.yaml`

## Key Technical Decisions
1. Polars-based feature engineering for efficient data transformation
2. Modular transformer system for organizing feature computation
3. DVC integration for reproducible ML pipelines
4. Registry pattern for dynamic feature loading
5. Pipeline pattern for orchestrating transformations

## Design Patterns
1. Transformer Pattern: Used in feature engineering for modular feature computation
2. Registry Pattern: Manages feature transformer registration and retrieval
3. Factory Pattern: Creates different types of features and models
4. Pipeline Pattern: Manages the flow from data collection to prediction

## Feature Engineering System

### Core Components
1. FeatureTransformer:
   - Base class for all transformers
   - Handles validation and common operations
   - Provides consistent interface

2. Feature Registry:
   - Central registry for transformer registration
   - Dynamic transformer loading
   - Configuration-driven feature selection

3. Data Loaders:
   - Abstract interface for data loading
   - Specialized implementations for different sources
   - Consistent data format handling

4. Feature Pipeline:
   - Orchestrates transformer execution
   - Handles data flow between transformers
   - Manages feature combination

### Feature Categories

#### Basic Statistics
- Minutes played and basic counting stats
- Shooting percentages and efficiency metrics
- PIR components and calculations

#### Rolling Statistics
- Moving averages over game windows
- Form and consistency metrics
- Performance trends

#### Shot Patterns
- Shot quality metrics
- Court coverage analysis
- Shot type distributions

#### Play-by-Play Analysis
- Game flow features
- Clutch performance metrics
- Usage patterns

#### Position Matchups
- Position-specific performance
- Matchup edge calculations
- Dominance rates

## Model Insights

### Feature Importance Analysis
Based on the latest feature importance analysis, the model shows clear patterns in feature utilization:

#### High Impact Features
1. Points (3.46): Strongest predictor of performance
2. FG percentage (0.66): Second most influential feature
3. Improving form (0.29): Moderate importance
4. Is_starter (0.18): Notable impact on predictions
5. Assist-to-turnover ratio (0.16): Minor but positive influence

#### Underutilized Features
Several advanced metrics currently show zero importance:
- Game context metrics (is_home, close_game_plays)
- Efficiency indicators (lineup_net_rating, primary_lineup_efficiency)
- Shot quality metrics (defender_proximity, shot_quality_score)
- Performance comparisons (pir_vs_expected, points_vs_expected)

#### Negative Impact Features
Time-based metrics showing inverse relationships:
- Points moving average (points_ma3: -0.62)
- Minutes played (-0.30)
- PIR moving average (pir_ma3: -0.25)
- PIR vs season average (-0.22)

### Improvement Recommendations

#### Feature Engineering
1. Create interaction terms:
   - Combine high-impact features (Points × FG_percentage)
   - Develop composite metrics from zero-importance features
   - Normalize features for consistent scaling

#### Feature Selection
1. Optimize feature set:
   - Remove or combine redundant PIR-related metrics
   - Investigate zero-importance advanced metrics
   - Apply LASSO or elastic net for dimensionality reduction

#### Domain-Specific Enhancements
1. Develop sophisticated indicators:
   - Enhanced form metrics beyond moving averages
   - Matchup-specific features incorporating defender stats
   - Temporal features for season progression

#### Model Architecture
1. Structural improvements:
   - Context-specific model variants
   - Non-linear relationship handling
   - Regularization to balance feature importance

#### Data Quality
1. Validation and preprocessing:
   - Verify zero-importance features
   - Implement consistent scaling
   - Consider feature binning for continuous variables