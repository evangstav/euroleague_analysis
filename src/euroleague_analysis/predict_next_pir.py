"""Predict next game PIR for all active Euroleague players."""

import json
import logging
from pathlib import Path
from typing import Any, Tuple

import joblib
import pandas as pd
import polars as pl

from .feature_engineering.pipeline import FeaturePipeline
from .model_config import FEATURE_COLUMNS, prepare_features

logger = logging.getLogger(__name__)


def load_model_artifacts(model_dir: Path = Path("models")) -> Tuple[Any, Any]:
    """Load trained model and scaler from specified directory.

    Args:
        model_dir: Path to directory containing model artifacts

    Returns:
        Tuple containing (model, scaler) objects

    Raises:
        FileNotFoundError: If model artifacts are not found
        RuntimeError: If there are issues loading the artifacts
    """
    try:
        model_path = model_dir / "pir_predictor.joblib"
        scaler_path = model_dir / "scaler.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}\n"
                "Please ensure the model has been trained and saved correctly."
            )

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found at {scaler_path}\n"
                "Please ensure the scaler has been saved during training."
            )

        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)

        logger.info("Successfully loaded model artifacts")
        return model, scaler

    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise RuntimeError("Failed to load model artifacts") from e


def get_latest_features(data_dir: Path = Path("euroleague_data")) -> pl.DataFrame:
    """Load latest features from feature engineering pipeline output.

    Args:
        data_dir: Path to directory containing feature data

    Returns:
        Polars DataFrame containing latest features

    Raises:
        FileNotFoundError: If features file is not found
        RuntimeError: If there are issues loading the features
    """
    try:
        features_path = data_dir / "features.parquet"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found at {features_path}\n"
                "Please run 'dvc repro' to generate features."
            )

        logger.info(f"Loading features from {features_path}")
        features_df = pl.read_parquet(features_path)

        logger.info(
            f"Loaded {len(features_df)} player records with {len(features_df.columns)} features"
        )
        logger.debug(f"Feature columns: {features_df.columns}")

        return features_df

    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        raise RuntimeError("Failed to load features") from e


from sklearn.inspection import permutation_importance


def get_feature_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Get feature importance information from the model using permutation importance."""
    try:
        # Calculate permutation importance
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42)

        # Create DataFrame with mean importance scores
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": r.importances_mean}
        )
        return importance_df.sort_values("importance", ascending=False)
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return pd.DataFrame()


def predict_next_pir(
    model: Any,
    scaler: Any,
    features_df: pl.DataFrame,
    season: int = 2024,
    data_dir: Path = Path("euroleague_data"),
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Generate PIR predictions for all players in specified season.

    Args:
        model: Trained prediction model
        scaler: Feature scaler
        features_df: DataFrame containing player features
        season: Season to predict for
        data_dir: Directory containing player stats data

    Returns:
        Tuple containing:
        - DataFrame with predictions and metadata
        - DataFrame with feature importance scores

    Raises:
        FileNotFoundError: If player stats data is not found
        RuntimeError: If there are issues generating predictions
    """
    try:
        # Get the latest round
        latest_round = (
            features_df.filter(pl.col("Season") == season).select("Round").max().item()
        )
        next_round = latest_round + 1
        logger.info(f"Generating predictions for Round {next_round}")

        # Get players from the latest round, excluding team totals
        latest_players = (
            features_df.filter(
                (pl.col("Season") == season)
                & (pl.col("Round") == latest_round)
                & (pl.col("Player_ID") != "Total")
            )
            .select("Player_ID")
            .unique()
            .to_series()
            .to_list()
        )
        logger.info(f"Found {len(latest_players)} players in Round {latest_round}")

        # Get latest data for each player
        player_df = (
            features_df.filter(
                (pl.col("Player_ID").is_in(latest_players))
                & (pl.col("Season") == season)
            )
            .sort(["Player_ID", "Round"], descending=[False, True])
            .group_by("Player_ID")
            .first()
        )

        logger.info(f"Initial player DataFrame shape: {player_df.shape}")
        logger.debug(f"Initial player DataFrame columns: {player_df.columns}")

        # Convert to pandas for model preparation
        player_df_pd = player_df.to_pandas()

        # Prepare features for prediction
        player_df_pd = prepare_features(player_df_pd, for_prediction=True)
        logger.info(f"After preparation player DataFrame shape: {player_df_pd.shape}")
        logger.debug(
            f"After preparation player DataFrame columns: {player_df_pd.columns}"
        )

        # Get feature matrix
        X = player_df_pd[FEATURE_COLUMNS]
        logger.info(f"Feature matrix shape: {X.shape}")

        # Scale features
        X_scaled = scaler.transform(X)

        # Generate predictions
        predictions = model.predict(X_scaled)

        # Load player metadata
        player_stats_path = data_dir / "raw" / f"player_stats_{season}.parquet"
        if not player_stats_path.exists():
            raise FileNotFoundError(
                f"Player stats file not found at {player_stats_path}\n"
                "Please ensure the data pipeline has been run."
            )

        player_stats = pl.read_parquet(player_stats_path)
        player_meta = (
            player_stats.filter(
                (pl.col("Player_ID") != "Total")
                & (pl.col("Player_ID").is_in(latest_players))
            )
            .select(["Player_ID", "Player", "Team"])
            .unique()
            .to_pandas()
        )

        # Create results dataframe with predictions
        results_df = pd.DataFrame(
            {
                "Player_ID": player_df_pd["Player_ID"],
                "PredictedPIR": predictions,
                "CurrentPIR": player_df_pd["PIR"],
                "PIRTrend": player_df_pd["pir_ma3"],
                "MinutesPlayed": player_df_pd["minutes_played"],
                "Points": player_df_pd["Points"],
                "Round": next_round,
            }
        )

        # Add player metadata
        results_df = results_df.merge(player_meta, on="Player_ID", how="left")

        # Sort by predicted PIR
        results_df = results_df.sort_values("PredictedPIR", ascending=False)

        # Get feature importance using the current data
        X = player_df_pd[FEATURE_COLUMNS]
        y = player_df_pd["PIR"]
        importance_df = get_feature_importance(model, X, y)

        logger.info(f"Generated predictions for {len(results_df)} players")
        return results_df, importance_df

    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise RuntimeError("Failed to generate predictions") from e


def save_predictions(
    predictions_df: pd.DataFrame, importance_df: pd.DataFrame, output_dir: Path
) -> None:
    """Save predictions and feature importance to JSON and CSV formats.

    Args:
        predictions_df: DataFrame containing player predictions
        importance_df: DataFrame containing feature importance scores
        output_dir: Directory to save output files

    Raises:
        RuntimeError: If there are issues saving the files
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving predictions to {output_dir}")

        # Save predictions
        predictions_path = output_dir / "player_predictions"
        predictions_dict = predictions_df.to_dict(orient="records")

        # Save as JSON
        json_path = predictions_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(predictions_dict, f, indent=2)
        logger.info(f"Saved predictions to {json_path}")

        # Save as CSV
        csv_path = predictions_path.with_suffix(".csv")
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

        # Save feature importance if available
        if not importance_df.empty:
            importance_path = output_dir / "feature_importance"

            # Save as JSON
            importance_json_path = importance_path.with_suffix(".json")
            with open(importance_json_path, "w") as f:
                json.dump(importance_df.to_dict(orient="records"), f, indent=2)
            logger.info(f"Saved feature importance to {importance_json_path}")

            # Save as CSV
            importance_csv_path = importance_path.with_suffix(".csv")
            importance_df.to_csv(importance_csv_path, index=False)
            logger.info(f"Saved feature importance to {importance_csv_path}")

        logger.info("Successfully saved all prediction outputs")

    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise RuntimeError("Failed to save predictions") from e


def format_prediction_report(
    predictions_df: pd.DataFrame, importance_df: pd.DataFrame, top_n: int = 20
) -> str:
    """Format predictions and feature importance into a readable report.

    Args:
        predictions_df: DataFrame containing player predictions
        importance_df: DataFrame containing feature importance scores
        top_n: Number of top predictions to include in report

    Returns:
        Formatted string report

    Raises:
        ValueError: If predictions_df is empty
    """
    if predictions_df.empty:
        raise ValueError("Cannot format report - predictions DataFrame is empty")

    try:
        next_round = predictions_df["Round"].iloc[0]
        report = [
            f"\n=== Top {top_n} Players by Predicted PIR for Round {next_round} ===",
            "=" * 60,
            "",
        ]

        # Add feature importance section if available
        if not importance_df.empty:
            report.extend(["\n=== Top 5 Most Important Features ===", "=" * 35, ""])
            for _, row in importance_df.head(5).iterrows():
                report.append(f"{row['feature']}: {row['importance']:.3f}")
            report.extend(["", "=== Player Predictions ===", "=" * 25, ""])

        # Format player predictions
        for _, row in predictions_df.head(top_n).iterrows():
            report.extend(
                [
                    f"{row['Player']} ({row['Team']})",
                    f"  Predicted PIR: {row['PredictedPIR']:.1f}",
                    "",
                ]
            )

        return "\n".join(report)

    except Exception as e:
        logger.error(f"Error formatting prediction report: {str(e)}")
        raise RuntimeError("Failed to format prediction report") from e


def main() -> None:
    """Run the complete prediction pipeline.

    Raises:
        RuntimeError: If any stage of the pipeline fails
    """
    try:
        # Set paths
        model_dir = Path("models")
        data_dir = Path("euroleague_data")
        output_dir = data_dir / "predictions"

        logger.info("Starting prediction pipeline")

        # Load model artifacts
        logger.info("Loading model artifacts")
        model, scaler = load_model_artifacts(model_dir)

        # Get latest features
        logger.info("Loading latest features")
        features_df = get_latest_features(data_dir)

        # Generate predictions
        logger.info("Generating predictions")
        predictions_df, importance_df = predict_next_pir(
            model, scaler, features_df, season=2024
        )

        # Save results
        logger.info("Saving predictions")
        save_predictions(predictions_df, importance_df, output_dir)

        # Generate and print report
        logger.info("Formatting report")
        report = format_prediction_report(predictions_df, importance_df)
        print(report)

        logger.info("Prediction pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise RuntimeError("Prediction pipeline failed") from e


if __name__ == "__main__":
    main()
