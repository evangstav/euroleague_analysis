"""
Model training and evaluation for predicting next game PIR in Euroleague.
Uses DVC for pipeline management.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.preprocessing import StandardScaler

from .model_config import FEATURE_COLUMNS, TARGET_COLUMN, prepare_features

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

    def plot_learning_curves(self, X: pd.DataFrame, y: pd.Series, output_dir: Path):
        """Plot learning curves to detect overfitting."""
        logger.info("Generating learning curves...")

        X_scaled = self.scaler.transform(X)
        train_sizes, train_scores, val_scores = learning_curve(
            self.model,
            X_scaled,
            y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )

        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label="Training error")
        plt.plot(train_sizes, val_scores_mean, label="Validation error")
        plt.xlabel("Training examples")
        plt.ylabel("Mean Squared Error")
        plt.title("Learning Curves")
        plt.legend(loc="best")
        plt.grid(True)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "learning_curves.png")
        plt.close()

    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series, output_dir: Path):
        """Plot feature importance using permutation importance."""
        logger.info("Generating feature importance plot...")

        try:
            # Calculate permutation importance
            r = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)

            # Create DataFrame with mean importance scores
            importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": r.importances_mean}
            )
            importance_df = importance_df.sort_values("importance", ascending=True)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, y="feature", x="importance")
            plt.title("Feature Importance (Permutation)")
            plt.xlabel("Mean Importance")
            plt.tight_layout()
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "feature_importance.png")
            plt.close()

            # Save raw importance scores
            importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {e}")

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
        fold_metrics = []

        X_scaled = self.scaler.transform(X)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model on this fold
            fold_model = HistGradientBoostingRegressor(
                **{k: v for k, v in self.model.get_params().items() if k != "verbose"}
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
                    "fold": fold,
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
            "fold_metrics": fold_metrics,
        }

        return {
            "metrics": avg_metrics,
            "predictions": np.array(all_predictions),
            "actuals": np.array(all_actuals),
        }

    def plot_prediction_scatter(
        self, predictions: np.ndarray, actuals: np.ndarray, output_dir: Path
    ):
        """Plot predictions vs actuals scatter plot."""
        logger.info("Generating prediction scatter plot...")

        plt.figure(figsize=(10, 10))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], "r--")
        plt.xlabel("Actual PIR")
        plt.ylabel("Predicted PIR")
        plt.title("Predictions vs Actuals")
        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "prediction_scatter.png")
        plt.close()

    def plot_cv_scores(self, fold_metrics: list, output_dir: Path):
        """Plot cross-validation score distribution."""
        logger.info("Generating cross-validation score distribution plot...")

        metrics_df = pd.DataFrame(fold_metrics)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(["rmse", "mae", "r2"]):
            sns.boxplot(y=metrics_df[metric], ax=axes[i])
            axes[i].set_title(f"{metric.upper()} Distribution")

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "cv_scores_distribution.png")
        plt.close()


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

    # Create plots directory
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Initialize predictor
    predictor = PIRPredictor(params)

    # Load data
    data_path = data_dir / "features.parquet"
    df = pd.read_parquet(data_path)

    # Prepare features
    X, y, df_processed = predictor.prepare_training_data(df)

    # Train model
    predictor.train(X, y)

    # Generate learning curves
    predictor.plot_learning_curves(X, y, plots_dir)

    # Generate feature importance plot using scaled features
    X_scaled = predictor.scaler.transform(X)
    predictor.plot_feature_importance(X_scaled, y, plots_dir)

    # Evaluate model
    evaluation = predictor.evaluate(X, y, df_processed)

    # Generate prediction scatter plot
    predictor.plot_prediction_scatter(
        evaluation["predictions"], evaluation["actuals"], plots_dir
    )

    # Generate CV scores distribution plot
    predictor.plot_cv_scores(evaluation["metrics"]["fold_metrics"], plots_dir)

    # Save results
    metrics_file = model_dir / "metrics.json"
    model_file = model_dir / "pir_predictor.joblib"
    scaler_file = model_dir / "scaler.joblib"

    # Save metrics
    with open(metrics_file, "w") as f:
        json.dump(evaluation["metrics"], f, indent=2)

    # Save model and scaler
    joblib.dump(predictor.model, model_file)
    joblib.dump(predictor.scaler, scaler_file)

    # Print results
    logger.info("\nModel Performance Metrics:")
    logger.info(f"RMSE: {evaluation['metrics']['rmse']:.2f}")
    logger.info(f"MAE: {evaluation['metrics']['mae']:.2f}")
    logger.info(f"R2 Score: {evaluation['metrics']['r2']:.2f}")

    logger.info(f"\nModel artifacts and visualizations saved to {model_dir}")


if __name__ == "__main__":
    main()
