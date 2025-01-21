#!/bin/bash

# Log file setup
LOG_FILE="$HOME/euroleague_model_updates.log"
REPORTS_DIR="reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Navigate to project directory
cd /Users/estav/workspace/euroleauge_analysis

# Create reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

# Activate Python environment if it exists
if [ -f ".python-version" ]; then
    eval "$(pyenv init -)"
    pyenv activate $(cat .python-version)
fi

# Force rerun of data collection
log_message "Starting data collection..."
if dvc repro -f load_data; then
    log_message "Data collection completed successfully"
else
    log_message "Error: Data collection failed"
    exit 1
fi

# Run remaining pipeline stages
log_message "Running feature engineering and model training..."
if dvc repro; then
    log_message "Pipeline completed successfully"
else
    log_message "Error: Pipeline failed"
    exit 1
fi

# Generate predictions
log_message "Generating predictions..."
PREDICTION_OUTPUT=$(uv run python -m src.euroleague_analysis.predict_next_pir)
PREDICTION_STATUS=$?

# Create markdown report
REPORT_FILE="$REPORTS_DIR/report_${TIMESTAMP}.md"

{
    echo "# Euroleague Analysis Report"
    echo "Generated on: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "## Model Performance"
    echo "\`\`\`"
    cat models/metrics.json
    echo "\`\`\`"
    echo
    echo "## Predictions for Next Round"
    echo "\`\`\`"
    echo "$PREDICTION_OUTPUT"
    echo "\`\`\`"
    echo
    echo "## Data Collection Summary"
    echo "- Latest data collection completed: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- Model retrained with updated data"
    if [ -f "euroleague_data/predictions/feature_importance.json" ]; then
        echo
        echo "## Feature Importance"
        echo "\`\`\`"
        cat euroleague_data/predictions/feature_importance.json
        echo "\`\`\`"
    fi
} > "$REPORT_FILE"

log_message "Report generated: $REPORT_FILE"

# Check if prediction failed
if [ $PREDICTION_STATUS -ne 0 ]; then
    log_message "Error: Prediction generation failed"
    exit 1
fi

log_message "Update process completed"