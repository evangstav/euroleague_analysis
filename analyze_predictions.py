"""
Analyze model predictions and generate insights about model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from model_config import FEATURE_COLUMNS, TARGET_COLUMN, prepare_features, ID_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("analyze", {})


def calculate_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate detailed prediction metrics."""
    residuals = y_true - y_pred

    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mean_error": np.mean(residuals),
        "std_error": np.std(residuals),
        "median_error": np.median(residuals),
        "q1_error": np.percentile(residuals, 25),
        "q3_error": np.percentile(residuals, 75),
        "max_overpredict": np.min(residuals),
        "max_underpredict": np.max(residuals),
        "within_5_pir": np.mean(np.abs(residuals) <= 5) * 100,
        "within_10_pir": np.mean(np.abs(residuals) <= 10) * 100,
    }

    return metrics


def analyze_predictions_by_feature(df: pd.DataFrame, feature: str) -> dict:
    """Analyze prediction errors across different values of a feature."""
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Create bins for numerical features
    df_copy["value_bin"] = pd.qcut(df_copy[feature], q=5, duplicates="drop")

    # Calculate metrics for each bin
    metrics_by_bin = []
    for bin_label in df_copy["value_bin"].unique():
        bin_data = df_copy[df_copy["value_bin"] == bin_label]
        metrics = {
            "bin": f"{bin_label.left:.2f} to {bin_label.right:.2f}",
            "rmse": np.sqrt(
                mean_squared_error(bin_data["true_PIR"], bin_data["pred_PIR"])
            ),
            "mae": mean_absolute_error(bin_data["true_PIR"], bin_data["pred_PIR"]),
            "mean_error": np.mean(bin_data["true_PIR"] - bin_data["pred_PIR"]),
            "count": len(bin_data),
        }
        metrics_by_bin.append(metrics)

    return metrics_by_bin


def create_analysis_plots(df: pd.DataFrame, output_dir: Path):
    """Create various analysis plots."""
    # Set style
    # plt.style.use('seaborn')

    # 1. Actual vs Predicted scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["true_PIR"], df["pred_PIR"], alpha=0.5)
    plt.plot(
        [df["true_PIR"].min(), df["true_PIR"].max()],
        [df["true_PIR"].min(), df["true_PIR"].max()],
        "r--",
        label="Perfect Prediction",
    )
    plt.xlabel("Actual PIR")
    plt.ylabel("Predicted PIR")
    plt.title("Actual vs Predicted PIR")
    plt.legend()
    plt.savefig(output_dir / "actual_vs_predicted.png")
    plt.close()

    # 2. Residuals plot
    plt.figure(figsize=(10, 6))
    residuals = df["true_PIR"] - df["pred_PIR"]
    plt.scatter(df["pred_PIR"], residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted PIR")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted PIR")
    plt.savefig(output_dir / "residuals.png")
    plt.close()

    # 3. Prediction error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Errors")
    plt.savefig(output_dir / "error_distribution.png")
    plt.close()

    # 4. Error by actual PIR range
    plt.figure(figsize=(10, 6))
    df["pir_bin"] = pd.qcut(df["true_PIR"], q=10, duplicates="drop")
    error_by_pir = df.groupby("pir_bin")["pred_PIR"].apply(
        lambda x: np.sqrt(mean_squared_error(df.loc[x.index, "true_PIR"], x))
    )
    plt.plot(range(len(error_by_pir)), error_by_pir.values, marker="o")
    plt.xticks(
        range(len(error_by_pir)),
        [f"{x.left:.1f}-{x.right:.1f}" for x in error_by_pir.index],
        rotation=45,
    )
    plt.xlabel("Actual PIR Range")
    plt.ylabel("RMSE")
    plt.title("Prediction Error by PIR Range")
    plt.tight_layout()
    plt.savefig(output_dir / "error_by_pir_range.png")
    plt.close()


def analyze_important_features(df: pd.DataFrame, importance_file: Path) -> dict:
    """Analyze how the model performs across different values of important features."""
    # Load feature importance
    with open(importance_file) as f:
        feature_importance = json.load(f)

    # Get top features
    top_features = [f["feature"] for f in feature_importance[:5]]

    # Analyze each important feature
    feature_analysis = {}
    for feature in top_features:
        if feature in df.columns and df[feature].dtype in ["float64", "int64"]:
            feature_analysis[feature] = analyze_predictions_by_feature(df, feature)

    return feature_analysis


def main():
    # Load parameters
    params = load_params()

    # Get paths
    data_dir = Path(params.get("data_dir", "euroleague_data"))
    model_dir = Path(params.get("model_dir", "models"))
    analysis_dir = Path(params.get("analysis_dir", "analysis"))
    analysis_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_parquet(data_dir / "features.parquet")

    # Load model and scaler
    model = joblib.load(model_dir / "pir_predictor.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")

    # Prepare features (using same logic as in training)
    df = prepare_features(df)

    # Make predictions
    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Create analysis dataset
    analysis_df = pd.DataFrame({"true_PIR": df[TARGET_COLUMN], "pred_PIR": predictions})

    # Add original features for analysis
    analysis_df = pd.concat([analysis_df, df[FEATURE_COLUMNS], df[ID_COLUMNS]], axis=1)

    # Calculate overall metrics
    metrics = calculate_prediction_metrics(
        analysis_df["true_PIR"], analysis_df["pred_PIR"]
    )

    # Analyze important features
    feature_analysis = analyze_important_features(
        analysis_df, model_dir / "feature_importance.json"
    )

    # Create plots
    create_analysis_plots(analysis_df, analysis_dir)

    # Add player-specific analysis
    player_metrics = analysis_df.groupby("Player_ID", as_index=False).apply(
        lambda x: pd.Series(
            {
                "rmse": np.sqrt(mean_squared_error(x["true_PIR"], x["pred_PIR"])),
                "mae": mean_absolute_error(x["true_PIR"], x["pred_PIR"]),
                "mean_error": np.mean(x["true_PIR"] - x["pred_PIR"]),
                "games_predicted": len(x),
            }
        )
    )

    # Get top 10 most predicted players and their metrics
    top_players = player_metrics[player_metrics["games_predicted"] >= 10].sort_values(
        "rmse"
    )

    # Save analysis results
    analysis_results = {
        "overall_metrics": metrics,
        "feature_analysis": feature_analysis,
        "top_players": top_players.head(10).to_dict(orient="records"),
        "plots_generated": [
            "actual_vs_predicted.png",
            "residuals.png",
            "error_distribution.png",
            "error_by_pir_range.png",
        ],
    }

    with open(analysis_dir / "analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

    # Print key insights
    logger.info("\nModel Performance Analysis:")
    logger.info(f"Overall RMSE: {metrics['rmse']:.2f}")
    logger.info(f"Overall MAE: {metrics['mae']:.2f}")
    logger.info(f"R² Score: {metrics['r2']:.2f}")
    logger.info(f"Predictions within 5 PIR: {metrics['within_5_pir']:.1f}%")
    logger.info(f"Predictions within 10 PIR: {metrics['within_10_pir']:.1f}%")
    logger.info("\nTop 5 Players with Best Predictions (min 10 games):")
    print(top_players[["Player_ID", "rmse", "mae", "games_predicted"]].head().round(2))
    logger.info(f"\nAnalysis results and plots saved to {analysis_dir}")


if __name__ == "__main__":
    main()

