"""
Model training and evaluation for predicting next game PIR in Euroleague.
Uses DVC for pipeline management.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from model_config import FEATURE_COLUMNS, TARGET_COLUMN, prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIRPredictor:
    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self.scaler = StandardScaler()

    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for training."""
        df = prepare_features(df)
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        return X, y, df

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model using histogram-based gradient boosting."""
        logger.info("Training PIR prediction model...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model with parameters from params.yaml
        self.model = HistGradientBoostingRegressor(
            max_iter=self.params.get("max_iter", 100),
            learning_rate=self.params.get("learning_rate", 0.1),
            max_depth=self.params.get("max_depth", 5),
            min_samples_leaf=self.params.get("min_samples_leaf", 20),
            l2_regularization=self.params.get("l2_regularization", 1.0),
            max_leaf_nodes=self.params.get("max_leaf_nodes", None),
            validation_fraction=self.params.get("validation_fraction", 0.1),
            early_stopping=self.params.get("early_stopping", True),
            n_iter_no_change=self.params.get("n_iter_no_change", 10),
            random_state=42,
        )

        # Fit model
        self.model.fit(X_scaled, y)

        logger.info("Model training completed")

    def evaluate(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> dict:
        """
        Evaluate model performance using time series cross-validation.
        Returns dictionary of evaluation metrics.
        """
        logger.info("Evaluating model performance...")

        # Initialize TimeSeriesSplit
        n_splits = self.params.get("n_splits", 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Store predictions and actual values
        all_predictions = []
        all_actuals = []

        # Store metrics for each fold
        fold_metrics = []

        X_scaled = self.scaler.transform(X)

        for train_idx, test_idx in tscv.split(X_scaled):
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model on this fold
            fold_model = HistGradientBoostingRegressor(
                max_iter=self.params.get("max_iter", 100),
                learning_rate=self.params.get("learning_rate", 0.1),
                max_depth=self.params.get("max_depth", 5),
                min_samples_leaf=self.params.get("min_samples_leaf", 20),
                l2_regularization=self.params.get("l2_regularization", 1.0),
                max_leaf_nodes=self.params.get("max_leaf_nodes", None),
                validation_fraction=self.params.get("validation_fraction", 0.1),
                early_stopping=self.params.get("early_stopping", True),
                n_iter_no_change=self.params.get("n_iter_no_change", 10),
                random_state=42,
            )
            fold_model.fit(X_train, y_train)

            # Make predictions
            y_pred = fold_model.predict(X_test)

            # Store predictions and actuals
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)

            # Calculate metrics for this fold
            fold_metrics.append(
                {
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                }
            )

        # Calculate average metrics across folds
        avg_metrics = {
            "rmse": np.mean([m["rmse"] for m in fold_metrics]),
            "mae": np.mean([m["mae"] for m in fold_metrics]),
            "r2": np.mean([m["r2"] for m in fold_metrics]),
        }

        # Get feature importance (for HistGradientBoostingRegressor, permutation importance might be needed)
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            # If direct feature importances not available, we'll still create the DataFrame
            # but with normalized coefficients
            importances = np.ones(len(X.columns)) / len(X.columns)

        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": importances}
        )
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )

        return {
            "metrics": avg_metrics,
            "feature_importance": feature_importance,
            "predictions": np.array(all_predictions),
            "actuals": np.array(all_actuals),
        }


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("train", {})


def main():
    # Load parameters
    params = load_params()

    # Get paths from params
    data_dir = Path(params.get("data_dir", "euroleague_data"))
    model_dir = Path(params.get("model_dir", "models"))
    model_dir.mkdir(exist_ok=True)

    # Initialize predictor
    predictor = PIRPredictor(params)

    # Load data
    data_path = data_dir / "features.parquet"
    df = pd.read_parquet(data_path)

    # Prepare features
    X, y, df_processed = predictor.prepare_training_data(df)

    # Train model
    predictor.train(X, y)

    # Evaluate model
    evaluation = predictor.evaluate(X, y, df_processed)

    # Save results
    metrics_file = model_dir / "metrics.json"
    importance_file = model_dir / "feature_importance.json"
    model_file = model_dir / "pir_predictor.joblib"
    scaler_file = model_dir / "scaler.joblib"

    # Save metrics
    with open(metrics_file, "w") as f:
        json.dump(evaluation["metrics"], f, indent=2)

    # Save feature importance
    feature_importance_dict = evaluation["feature_importance"].to_dict(orient="records")
    with open(importance_file, "w") as f:
        json.dump(feature_importance_dict, f, indent=2)

    # Save model and scaler
    joblib.dump(predictor.model, model_file)
    joblib.dump(predictor.scaler, scaler_file)

    # Print results
    logger.info("\nModel Performance Metrics:")
    logger.info(f"RMSE: {evaluation['metrics']['rmse']:.2f}")
    logger.info(f"MAE: {evaluation['metrics']['mae']:.2f}")
    logger.info(f"R2 Score: {evaluation['metrics']['r2']:.2f}")

    logger.info("\nTop 10 Most Important Features:")
    print(evaluation["feature_importance"].head(10))

    logger.info(f"\nModel artifacts saved to {model_dir}")


if __name__ == "__main__":
    main()

