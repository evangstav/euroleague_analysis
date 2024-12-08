"""
Probabilistic Graphical Model for PIR prediction using PyMC.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import json
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("train_pgm", {})


def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare data for the Bayesian model."""
    logger.info("Preparing data...")

    # Sort by time
    df = df.sort_values(["Season", "Round"])

    # Create target (next game PIR)
    df["next_PIR"] = df.groupby("Player_ID")["PIR"].shift(-1)

    # Select core features
    features = [
        "PIR",
        "minutes_played",
        "Points",  # Core stats
        "pir_ma3",  # Rolling average (most important predictor)
    ]

    # Create player tiers using LabelEncoder
    le = LabelEncoder()
    df["player_tier_idx"] = le.fit_transform(df["player_tier"])
    n_tiers = len(le.classes_)

    logger.info(f"Number of player tiers: {n_tiers}")
    logger.info(f"Player tiers: {le.classes_}")

    # Remove rows with missing target or features
    mask = ~df["next_PIR"].isna() & df[features].notna().all(axis=1)
    df_clean = df[mask].copy()

    logger.info(f"Data shape after cleaning: {df_clean.shape}")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[features])
    y = (df_clean["next_PIR"].values - df_clean["next_PIR"].mean()) / df_clean[
        "next_PIR"
    ].std()  # Standardize target
    player_tiers = df_clean["player_tier_idx"].values

    # Ensure player tiers are 0-based continuous integers
    assert player_tiers.min() == 0
    assert player_tiers.max() == n_tiers - 1

    # Prepare coordinates
    coords = {
        "games": range(len(df_clean)),
        "features": features,
        "player_tiers": range(n_tiers),
    }

    # Verify no NaNs remain
    assert not np.any(np.isnan(X)), "NaN values found in features"
    assert not np.any(np.isnan(y)), "NaN values found in target"

    # Store target statistics for later un-standardization
    target_stats = {
        "mean": float(df_clean["next_PIR"].mean()),
        "std": float(df_clean["next_PIR"].std()),
    }

    return X, y, player_tiers, coords, scaler, target_stats


def build_model(
    X: np.ndarray, y: np.ndarray, player_tiers: np.ndarray, coords: dict
) -> pm.Model:
    """Build the generative model for PIR prediction."""
    logger.info("Building model...")

    with pm.Model(coords=coords) as model:
        # Prior scale for regularization
        α = pm.HalfNormal("α", sigma=1)

        # Global coefficients
        β = pm.Normal("β", mu=0, sigma=α, dims="features")

        # Tier-specific intercepts
        tier_offset = pm.Normal("tier_offset", mu=0, sigma=0.5, dims="player_tiers")

        # Error term
        σ = pm.HalfNormal("σ", sigma=1)

        # Linear predictor
        μ = pm.math.dot(X, β) + tier_offset[player_tiers]

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=μ, sigma=σ, observed=y)

    return model


def train_model(model: pm.Model, params: dict) -> az.InferenceData:
    """Train the model using MCMC."""
    logger.info("Training model...")

    with model:
        # Sample from posterior
        idata = pm.sample(
            draws=params.get("n_samples", 1000),
            tune=params.get("n_tune", 1000),
            chains=params.get("n_chains", 2),  # Reduced number of chains
            cores=1,  # Single core to avoid multiprocessing issues
            random_seed=42,
            return_inferencedata=True,
        )
    return idata


def predict(
    model: pm.Model,
    idata: az.InferenceData,
    X: np.ndarray,
    player_tiers: np.ndarray,
    target_stats: dict,
) -> dict:
    """Make predictions with the trained model."""
    logger.info("Making predictions...")

    # Extract posterior samples
    β_samples = idata.posterior["β"].values
    tier_offset_samples = idata.posterior["tier_offset"].values
    σ_samples = idata.posterior["σ"].values

    # Calculate predictions
    n_samples = β_samples.shape[0] * β_samples.shape[1]
    predictions = np.zeros((len(X), n_samples))

    for i in range(β_samples.shape[0]):
        for j in range(β_samples.shape[1]):
            idx = i * β_samples.shape[1] + j
            μ = np.dot(X, β_samples[i, j]) + tier_offset_samples[i, j, player_tiers]
            predictions[:, idx] = μ

    # Un-standardize predictions
    predictions = predictions * target_stats["std"] + target_stats["mean"]

    # Calculate summary statistics
    pred_mean = predictions.mean(axis=1)
    pred_std = predictions.std(axis=1)
    pred_lower = np.percentile(predictions, 2.5, axis=1)
    pred_upper = np.percentile(predictions, 97.5, axis=1)

    return {
        "mean": pred_mean,
        "std": pred_std,
        "lower": pred_lower,
        "upper": pred_upper,
    }


def evaluate_model(predictions: dict, y: np.ndarray, target_stats: dict) -> dict:
    """Evaluate model performance."""
    logger.info("Evaluating model...")

    # Un-standardize true values
    y_true = y * target_stats["std"] + target_stats["mean"]

    # Calculate metrics
    metrics = {
        "rmse": float(np.sqrt(np.mean((y_true - predictions["mean"]) ** 2))),
        "mae": float(np.mean(np.abs(y_true - predictions["mean"]))),
        "coverage_95": float(
            np.mean((y_true >= predictions["lower"]) & (y_true <= predictions["upper"]))
        ),
        "avg_interval_width": float(
            np.mean(predictions["upper"] - predictions["lower"])
        ),
    }

    return metrics


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
    X, y, player_tiers, coords, scaler, target_stats = prepare_data(df)

    # Build and train model
    model = build_model(X, y, player_tiers, coords)
    idata = train_model(model, params)

    # Make predictions
    predictions = predict(model, idata, X, player_tiers, target_stats)

    # Evaluate model
    metrics = evaluate_model(predictions, y, target_stats)

    # Save results
    logger.info("Saving results...")

    # Save metrics
    with open(model_dir / "pgm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save parameter summaries
    summary = az.summary(idata)
    with open(model_dir / "pgm_parameters.json", "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    # Save model artifacts
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(target_stats, model_dir / "target_stats.joblib")
    az.to_netcdf(idata, model_dir / "inference_data.nc")

    # Print results
    logger.info("\nModel Performance:")
    logger.info(f"RMSE: {metrics['rmse']:.2f}")
    logger.info(f"MAE: {metrics['mae']:.2f}")
    logger.info(f"95% CI Coverage: {metrics['coverage_95']:.2f}")
    logger.info(f"Avg. Interval Width: {metrics['avg_interval_width']:.2f}")


if __name__ == "__main__":
    main()

