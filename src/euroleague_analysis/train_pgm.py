"""
Enhanced Probabilistic Graphical Model for PIR prediction using PyMC.
"""

import json
import logging
import warnings
from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pandas as pd
import pymc as pm
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("train_pgm", {})


def generate_predictions(
    model: pm.Model,
    idata: az.InferenceData,
    X_dict: dict,
    player_tiers: np.ndarray,
    player_ids: np.ndarray,
    target_stats: dict,
) -> dict:
    """Generate predictions from the trained model."""
    logger.info("Generating predictions...")

    with model:
        post_pred = pm.sample_posterior_predictive(
            idata, var_names=["y_obs"], random_seed=42
        )

    # Extract predictions and un-standardize
    preds = post_pred.posterior_predictive["y_obs"]
    preds = preds * target_stats["std"] + target_stats["mean"]

    # Calculate summary statistics
    pred_mean = preds.mean(dim=["chain", "draw"]).values
    pred_std = preds.std(dim=["chain", "draw"]).values
    pred_lower = preds.quantile(0.025, dim=["chain", "draw"]).values
    pred_upper = preds.quantile(0.975, dim=["chain", "draw"]).values

    return {
        "mean": pred_mean,
        "std": pred_std,
        "lower": pred_lower,
        "upper": pred_upper,
    }


def evaluate_predictions(predictions: dict, y: np.ndarray, target_stats: dict) -> dict:
    """Evaluate model predictions."""
    logger.info("Evaluating predictions...")

    # Un-standardize true values
    y_true = y * target_stats["std"] + target_stats["mean"]

    metrics = {
        "rmse": float(np.sqrt(np.mean((y_true - predictions["mean"]) ** 2))),
        "mae": float(np.mean(np.abs(y_true - predictions["mean"]))),
        "coverage_95": float(
            np.mean((y_true >= predictions["lower"]) & (y_true <= predictions["upper"]))
        ),
        "avg_interval_width": float(
            np.mean(predictions["upper"] - predictions["lower"])
        ),
        "std_interval_width": float(
            np.std(predictions["upper"] - predictions["lower"])
        ),
    }

    return metrics


def analyze_feature_importance(idata: az.InferenceData, metadata: dict) -> dict:
    """Analyze feature importance from the model."""
    logger.info("Analyzing feature importance...")

    importance_dict = {}

    for group in metadata["features"].keys():
        group_summary = az.summary(idata, var_names=[f"{group}_β"])
        features = metadata["features"][group]

        importance_dict[group] = {
            "features": features,
            "effects": group_summary["mean"].tolist(),
            "uncertainties": group_summary["sd"].tolist(),
        }

    return importance_dict


def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare data for the Bayesian model."""
    logger.info("Preparing data...")

    # Sort by time
    df = df.sort_values(["Season", "Round"])

    # Create target (next game PIR)
    df["next_PIR"] = df.groupby("Player_ID")["PIR"].shift(-1)

    # Select features
    features = {
        "performance": ["PIR", "minutes_played", "Points"],
        "efficiency": ["fg_percentage", "ft_percentage", "ast_to_turnover"],
        "momentum": ["pir_ma3", "points_ma3", "minutes_ma3"],
        "context": ["is_starter", "is_home"],
    }
    all_features = [f for group in features.values() for f in group]

    # Remove rows with missing target or features first
    mask = ~df["next_PIR"].isna() & df[all_features].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Now create encoders after cleaning
    player_encoder = LabelEncoder()
    df_clean["player_idx"] = player_encoder.fit_transform(df_clean["Player_ID"])

    tier_encoder = LabelEncoder()
    df_clean["tier_idx"] = tier_encoder.fit_transform(df_clean["player_tier"])

    logger.info(f"Data shape: {df_clean.shape}")
    logger.info(f"Number of unique players: {len(player_encoder.classes_)}")
    logger.info(f"Number of tiers: {len(tier_encoder.classes_)}")

    # Scale features by group
    X_dict = {}
    scalers = {}
    for group, group_features in features.items():
        scaler = StandardScaler()
        X_dict[group] = scaler.fit_transform(df_clean[group_features])
        scalers[group] = scaler

    # Standardize target
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(df_clean[["next_PIR"]]).ravel()

    # Store target statistics
    target_stats = {"mean": float(y_scaler.mean_[0]), "std": float(y_scaler.scale_[0])}

    metadata = {
        "features": features,
        "all_features": all_features,
        "n_features": {group: len(feats) for group, feats in features.items()},
        "n_players": len(player_encoder.classes_),
        "n_tiers": len(tier_encoder.classes_),
        "player_map": dict(enumerate(player_encoder.classes_)),
        "tier_map": dict(enumerate(tier_encoder.classes_)),
    }

    return (
        X_dict,
        y,
        df_clean["tier_idx"].values,
        df_clean["player_idx"].values,
        metadata,
        scalers,
        target_stats,
    )


def build_and_train_model(
    X_dict: dict,
    y: np.ndarray,
    tier_indices: np.ndarray,
    player_indices: np.ndarray,
    metadata: dict,
    params: dict,
) -> tuple:
    """Build and train the enhanced model."""
    logger.info("Building and training model...")

    with pm.Model() as model:
        # Global parameters
        global_σ = pm.HalfNormal("global_σ", sigma=2)

        # Group-specific parameters
        group_effects = {}
        for group, X_group in X_dict.items():
            # Group-level shrinkage
            group_σ = pm.HalfNormal(f"{group}_σ", sigma=global_σ)

            # Feature coefficients within group
            group_effects[group] = pm.Normal(
                f"{group}_β", mu=0, sigma=group_σ, shape=X_group.shape[1]
            )

        # Player tier effects
        tier_σ = pm.HalfNormal("tier_σ", sigma=1)
        tier_effects = pm.Normal(
            "tier_effects", mu=0, sigma=tier_σ, shape=metadata["n_tiers"]
        )

        # Player-specific random effects
        player_σ = pm.HalfNormal("player_σ", sigma=0.5)
        player_effects = pm.Normal(
            "player_effects", mu=0, sigma=player_σ, shape=metadata["n_players"]
        )

        # Combine all effects
        μ = 0
        # Add group effects
        for group, X_group in X_dict.items():
            μ += pm.math.dot(X_group, group_effects[group])

        # Add tier and player effects
        μ += tier_effects[tier_indices]
        μ += player_effects[player_indices]

        # Observation noise
        σ_obs = pm.HalfNormal("σ_obs", sigma=1)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=μ, sigma=σ_obs, observed=y)

        # Sample from posterior
        logger.info("Sampling from posterior...")
        idata = pm.sample(
            draws=params.get("n_samples", 1000),
            tune=params.get("n_tune", 1000),
            chains=2,
            cores=1,
            return_inferencedata=True,
            target_accept=0.95,
        )

    return model, idata


def main():
    # Load parameters
    params = load_params()

    # Set up paths
    data_dir = Path(params.get("data_dir", "euroleague_data"))
    model_dir = Path(params.get("model_dir", "models_pgm"))
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_parquet(data_dir / "features.parquet")

    # Prepare data
    X_dict, y, tier_indices, player_indices, metadata, scalers, target_stats = (
        prepare_data(df)
    )

    # Build and train model
    model, idata = build_and_train_model(
        X_dict, y, tier_indices, player_indices, metadata, params
    )

    # Generate predictions
    predictions = generate_predictions(
        model, idata, X_dict, tier_indices, player_indices, target_stats
    )

    # Evaluate model
    metrics = evaluate_predictions(predictions, y, target_stats)

    # Analyze feature importance
    importance = analyze_feature_importance(idata, metadata)

    # Save results
    logger.info("Saving results...")

    # Save metrics
    with open(model_dir / "pgm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save parameter summaries
    summary = az.summary(idata)
    with open(model_dir / "pgm_parameters.json", "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    # Save feature importance
    with open(model_dir / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    # Save model artifacts
    joblib.dump(scalers, model_dir / "scalers.joblib")
    joblib.dump(target_stats, model_dir / "target_stats.joblib")
    joblib.dump(metadata, model_dir / "metadata.joblib")
    az.to_netcdf(idata, model_dir / "inference_data.nc")

    # Print results
    logger.info("\nModel Performance:")
    logger.info(f"RMSE: {metrics['rmse']:.2f}")
    logger.info(f"MAE: {metrics['mae']:.2f}")
    logger.info(f"95% CI Coverage: {metrics['coverage_95']:.2f}")
    logger.info(f"Avg. Interval Width: {metrics['avg_interval_width']:.2f}")

    # Print feature importance by group
    logger.info("\nFeature Importance by Group:")
    for group, group_info in importance.items():
        logger.info(f"\n{group.upper()} Features:")
        for feat, effect, uncert in zip(
            group_info["features"], group_info["effects"], group_info["uncertainties"]
        ):
            logger.info(f"{feat}: {effect:.3f} ± {uncert:.3f}")

    # Print some player effect statistics
    player_effects_summary = az.summary(idata, var_names=["player_effects"])
    # Convert string indices to integers
    player_effects_summary.index = [
        int(idx.split("[")[-1].strip("]")) for idx in player_effects_summary.index
    ]

    top_players = player_effects_summary.nlargest(5, "mean")
    logger.info("\nTop 5 Player Effects:")
    for idx, row in top_players.iterrows():
        player_id = metadata["player_map"][idx]
        logger.info(f"Player {player_id}: {row['mean']:.3f} ± {row['sd']:.3f}")


if __name__ == "__main__":
    main()
