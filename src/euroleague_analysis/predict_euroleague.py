"""
Script to generate predictions for upcoming Euroleague matches.
"""

import logging
import joblib
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from predict_next_matches import generate_matchup_features
from model_config import FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str = "models/pir_predictor.pkl") -> Any:
    """Load the trained model from disk."""
    try:
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def prepare_player_features(player_data: Dict) -> pd.Series:
    """Prepare player features for prediction."""
    features = pd.Series(index=FEATURE_COLUMNS, dtype=float)

    # Map available features
    feature_mapping = {
        "avg_minutes": "minutes_played",
        "avg_pir": "PIR",
        "avg_points": "Points",
        "avg_fg_pct": "fg_percentage",
        "avg_ast_to": "ast_to_turnover",
        "is_starter": "is_starter",
        "pir_trend": "pir_ma3",
        "points_trend": "points_ma3",
    }

    # Fill known features
    for source, target in feature_mapping.items():
        if source in player_data:
            features[target] = player_data[source]

    # Fill missing features with 0
    features = features.fillna(0)

    return features


def predict_match(matchup_features: Dict, model: Any) -> Dict[str, Any]:
    """Generate predictions for a single match."""

    home_team, away_team = next(iter(matchup_features)).split("_vs_")
    features = matchup_features[f"{home_team}_vs_{away_team}"]

    if "error" in features:
        return {"error": features["error"]}

    # Process home team
    home_predictions = []
    for player in features["player_stats"]["home_team"]:
        player_features = prepare_player_features(player)
        prediction = model.predict([player_features])[0]
        home_predictions.append(
            {
                "PlayerName": player["PlayerName"],
                "PredictedPIR": prediction,
                "RecentPIR": player["avg_pir"],
                "Form": player["pir_trend"],
                "GamesPlayed": player["games_played"],
            }
        )

    # Process away team
    away_predictions = []
    for player in features["player_stats"]["away_team"]:
        player_features = prepare_player_features(player)
        prediction = model.predict([player_features])[0]
        away_predictions.append(
            {
                "PlayerName": player["PlayerName"],
                "PredictedPIR": prediction,
                "RecentPIR": player["avg_pir"],
                "Form": player["pir_trend"],
                "GamesPlayed": player["games_played"],
            }
        )

    # Sort predictions by predicted PIR
    home_predictions = sorted(
        home_predictions, key=lambda x: x["PredictedPIR"], reverse=True
    )
    away_predictions = sorted(
        away_predictions, key=lambda x: x["PredictedPIR"], reverse=True
    )

    # Calculate team totals
    home_team_pir = sum(p["PredictedPIR"] for p in home_predictions)
    away_team_pir = sum(p["PredictedPIR"] for p in away_predictions)

    # Add home court advantage (3% boost)
    home_team_pir *= 1.03

    # Calculate win probability
    total_pir = home_team_pir + away_team_pir
    win_prob = home_team_pir / total_pir if total_pir > 0 else 0.5

    return {
        "match_info": {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": win_prob,
        },
        "predictions": {
            "home_team": home_predictions,
            "away_team": away_predictions,
            "team_totals": {
                "home_team_pir": home_team_pir,
                "away_team_pir": away_team_pir,
            },
        },
    }


def generate_predictions(
    matchups: List[Tuple[str, str]],
    data_dir: Path = Path("euroleague_data"),
    season: int = 2024,
    model_path: str = "models/pir_predictor.pkl",
) -> Dict[str, Dict]:
    """Generate predictions for multiple matches."""

    # Load model
    model = load_model(model_path)

    season = 2024

    # Get features
    matchup_features = generate_matchup_features(matchups, data_dir, season)

    # Generate predictions
    predictions = {}
    for matchup in matchups:
        home_team, away_team = matchup
        key = f"{home_team}_vs_{away_team}"

        if key not in matchup_features:
            logger.warning(f"No features found for {key}")
            continue

        predictions[key] = predict_match({key: matchup_features[key]}, model)

    return predictions


def format_prediction_report(predictions: Dict[str, Dict]) -> str:
    """Format predictions into a readable report."""
    report = []

    for matchup, data in predictions.items():
        if "error" in data:
            report.append(f"\n=== {matchup} ===")
            report.append(f"Error: {data['error']}")
            continue

        match_info = data["match_info"]
        report.append(
            f"\n=== {match_info['home_team']} vs {match_info['away_team']} ==="
        )
        report.append(
            f"Win Probability: {match_info['home_team']}: {match_info['home_win_probability']:.1%}"
        )

        # Home team predictions
        report.append(f"\n{match_info['home_team']} Predictions:")
        for player in data["predictions"]["home_team"][:5]:  # Top 5 players
            report.append(
                f"- {player['PlayerName']}: "
                f"PIR {player['PredictedPIR']:.1f} "
                f"(Recent: {player['RecentPIR']:.1f}, Form: {player['Form']:.1f})"
            )

        # Away team predictions
        report.append(f"\n{match_info['away_team']} Predictions:")
        for player in data["predictions"]["away_team"][:5]:  # Top 5 players
            report.append(
                f"- {player['PlayerName']}: "
                f"PIR {player['PredictedPIR']:.1f} "
                f"(Recent: {player['RecentPIR']:.1f}, Form: {player['Form']:.1f})"
            )

        # Team totals
        totals = data["predictions"]["team_totals"]
        report.append(f"\nTeam Totals:")
        report.append(f"{match_info['home_team']}: {totals['home_team_pir']:.1f} PIR")
        report.append(f"{match_info['away_team']}: {totals['away_team_pir']:.1f} PIR")
        report.append("\n" + "=" * 50)

    return "\n".join(report)


def main():
    # Example matchups
    matchups = [
        ("Real Madrid", "Barcelona"),
        ("Olympiacos", "Panathinaikos"),
        ("Fenerbahce", "Anadolu Efes"),
    ]

    # Set paths
    data_dir = Path("euroleague_data")
    model_path = "models/pir_predictor.joblib"

    try:
        # Generate predictions
        predictions = generate_predictions(
            matchups, data_dir=data_dir, model_path=model_path
        )

        # Generate and print report
        report = format_prediction_report(predictions)
        print(report)

        # Save predictions to file
        output_file = data_dir / "predictions" / "latest_predictions.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Predictions saved to {output_file}")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
