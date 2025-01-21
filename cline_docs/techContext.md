# Technical Context

## Technologies Used

### Core Technologies
- Python: Primary programming language
- DVC (Data Version Control): For managing data and model versions
- SQL: For data transformation and feature engineering

### Development Tools
- Pre-commit hooks: Code quality checks
- UV: Python package manager
- VSCode: Primary development environment

### Key Dependencies
Based on requirements.txt and pyproject.toml:
- Data processing and analysis libraries
- Machine learning frameworks
- Database management tools

## Development Setup
1. Python environment management:
   - Uses `.python-version` for version control
   - UV for dependency management
   - Requirements specified in `requirements.txt` and `pyproject.toml`

2. Version Control:
   - Git for code version control
   - DVC for data and model versioning
   - Pre-commit hooks for code quality

3. Project Structure:
   ```
   ├── analyse_datasets/     # Analysis scripts
   ├── models/              # Model artifacts
   ├── reports/             # Generated reports
   ├── src/                 # Main source code
   │   └── euroleague_analysis/
   │       ├── feature_engineering/
   │       └── sql_queries/
   └── tests/               # Test suite
   ```

## Technical Constraints
1. Data Management:
   - Must handle large datasets efficiently
   - Requires version control for both data and models
   - SQL-based transformations for performance

2. Model Development:
   - Reproducible training pipeline
   - Version controlled model artifacts
   - Configurable model parameters

3. Code Quality:
   - Pre-commit hooks enforce standards
   - Python type hints and documentation
   - Modular architecture for maintainability