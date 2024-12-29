"""
Script to predict next game PIR for all active Euroleague players.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from .feature_engineering.builder import FeatureBuilder
from .feature_engineering.config import FeatureConfig
from .model_config import FEATURE_COLUMNS, prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_artifacts(model_dir: Path = Path("models")) -> tuple:
    """Load the trained model and scaler."""
    try:
        model = joblib.load(model_dir / "pir_predictor.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        logger.info("Successfully loaded model artifacts")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise


def get_latest_features(data_dir: Path = Path("euroleague_data")) -> pd.DataFrame:
    """Get latest features from DVC pipeline output."""
    try:
        # Load features from parquet file (output of feature engineering stage)
        features_path = data_dir / "features.parquet"
        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found at {features_path}. "
                "Please run 'dvc repro' to generate features."
            )

        # Load and prepare features
        features_df = pd.read_parquet(features_path)
        features_df = prepare_features(features_df)

        logger.info(f"Loaded features for {len(features_df)} player records")
        return features_df
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        raise


def predict_next_pir(model, scaler, features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate PIR predictions for all players."""
    try:
        # Filter out team totals
        player_df = features_df[features_df["Player_ID"] != "Total"].copy()

        # Load player metadata
        player_stats = pd.read_parquet("euroleague_data/raw/player_stats_2023.parquet")
        player_meta = player_stats[["Player_ID", "Player", "Team"]].drop_duplicates()

        # Get feature matrix
        X = player_df[FEATURE_COLUMNS]

        # Scale features
        X_scaled = scaler.transform(X)

        # Generate predictions
        predictions = model.predict(X_scaled)

        # Create results dataframe with predictions
        results_df = pd.DataFrame(
            {
                "Player_ID": player_df["Player_ID"],
                "PredictedPIR": predictions,
                "CurrentPIR": player_df["PIR"],
                "PIRTrend": player_df["pir_ma3"],
                "MinutesPlayed": player_df["minutes_played"],
                "Points": player_df["Points"],
                "IsStarter": player_df["is_starter"],
                "IsHome": player_df["is_home"],
                "Round": player_df["Round"],
            }
        )

        # Add player metadata
        results_df = results_df.merge(player_meta, on="Player_ID", how="left")

        # Sort by predicted PIR and Round
        results_df = results_df.sort_values(
            ["PredictedPIR", "Round"], ascending=[False, True]
        )

        logger.info(f"Generated predictions for {len(results_df)} players")
        return results_df
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise


def save_predictions(predictions_df: pd.DataFrame, output_dir: Path):
    """Save predictions to JSON and CSV formats."""
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        predictions_dict = predictions_df.to_dict(orient="records")
        json_path = output_dir / "player_predictions.json"
        with open(json_path, "w") as f:
            json.dump(predictions_dict, f, indent=2)

        # Save as CSV
        csv_path = output_dir / "player_predictions.csv"
        predictions_df.to_csv(csv_path, index=False)

        logger.info(f"Saved predictions to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise


def format_prediction_report(predictions_df: pd.DataFrame, top_n: int = 20) -> str:
    """Format top predictions into a readable report."""
    report = ["\n=== Top Players by Predicted PIR ===\n"]

    for _, row in predictions_df.head(top_n).iterrows():
        report.append(
            f"{row['Player']} ({row['Team']})\n"
            f"  Round: {row['Round']}\n"
            f"  Predicted PIR: {row['PredictedPIR']:.1f}\n"
            f"  Current PIR: {row['CurrentPIR']:.1f}\n"
            f"  PIR Trend: {row['PIRTrend']:.1f}\n"
            f"  Minutes: {row['MinutesPlayed']:.1f}\n"
            f"  Points: {row['Points']:.1f}\n"
            f"  Starter: {'Yes' if row['IsStarter'] else 'No'}\n"
            f"  Home: {'Yes' if row['IsHome'] else 'No'}\n"
            f"  Player ID: {row['Player_ID']}\n"
        )

    return "\n".join(report)


def main():
    # Set paths
    model_dir = Path("models")
    data_dir = Path("euroleague_data")
    output_dir = data_dir / "predictions"

    try:
        # Load model artifacts
        model, scaler = load_model_artifacts(model_dir)

        # Get latest features
        features_df = get_latest_features(data_dir)

        # Generate predictions
        predictions_df = predict_next_pir(model, scaler, features_df)

        # Save predictions
        save_predictions(predictions_df, output_dir)

        # Generate and print report
        report = format_prediction_report(predictions_df)
        print(report)

        logger.info("Prediction pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
