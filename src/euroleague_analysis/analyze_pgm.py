"""
Analysis of Probabilistic Graphical Model predictions for PIR.
"""

import json
import logging
from math import inf
from pathlib import Path
from typing import Counter

import arviz as az
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
import yaml
from pymc import HalfCauchy, Model, Normal, sample

print(f"Running on PyMC v{pm.__version__}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params.get("analyze_pgm", {})


def load_model_artifacts(model_dir: Path) -> dict:
    """Load all model artifacts."""
    logger.info("Loading model artifacts...")

    return {
        "inference_data": az.from_netcdf(model_dir / "inference_data.nc"),
        "metadata": joblib.load(model_dir / "metadata.joblib"),
        "target_stats": joblib.load(model_dir / "target_stats.joblib"),
        "metrics": json.load(open(model_dir / "pgm_metrics.json")),
        "parameters": json.load(open(model_dir / "pgm_parameters.json")),
        "scalers": joblib.load(model_dir / "scalers.joblib"),
    }


def reconstruct_predictions(
    idata: az.InferenceData,
    X_dict: dict,
    metadata: dict,
    tier_indices: np.ndarray,
    player_indices: np.ndarray,
) -> xr.DataArray:
    """Reconstruct model predictions from posterior samples."""
    logger.info("Reconstructing predictions from posterior samples...")

    # Initialize predictions as a writable DataArray
    chains = idata.posterior.dims["chain"]
    draws = idata.posterior.dims["draw"]
    y_model = xr.DataArray(
        np.zeros((chains, draws)),
        dims=("chain", "draw"),
        coords={"chain": idata.posterior.chain, "draw": idata.posterior.draw},
    )

    for group, X_group in X_dict.items():
        group_effect = idata.posterior[f"{group}_β"]  # Shape: (chain, draw, features)

        # **Corrected part:**
        X_group_array = xr.DataArray(
            X_group,
            dims=["sample", "feature"],  # More descriptive dimension names
            coords={
                "sample": np.arange(X_group.shape[0]),
                "feature": np.arange(X_group.shape[1]),
            },
        )
        # Align dimensions for the dot product
        result = xr.dot(group_effect, X_group_array, dims="feature")

        y_model, result = xr.broadcast(y_model, result)
        y_model = y_model.copy()
        y_model += result

    tier_effects = idata.posterior["tier_effects"]  # Shape: (chain, draw, tiers)

    tier_indices_array = xr.DataArray(
        tier_indices, dims=["sample"], coords={"sample": np.arange(len(tier_indices))}
    )

    tier_effect_contrib = tier_effects.isel(tier_effects_dim_0=tier_indices_array)

    y_model, tier_effect_contrib = xr.broadcast(y_model, tier_effect_contrib)
    y_model = y_model.copy()
    y_model += tier_effect_contrib

    player_effects = idata.posterior["player_effects"]  # Shape: (chain, draw, players)

    player_indices_array = xr.DataArray(
        player_indices,
        dims=["sample"],
        coords={"sample": np.arange(len(player_indices))},
    )
    player_effect_contrib = player_effects.isel(
        player_effects_dim_0=player_indices_array
    )
    y_model, player_effect_contrib = xr.broadcast(y_model, player_effect_contrib)
    y_model = y_model.copy()
    y_model += player_effect_contrib

    idata.posterior["y_model"] = y_model
    return idata


def analyze_posterior_distributions(idata: az.InferenceData, output_dir: Path):
    """Analyze and plot posterior distributions of key parameters."""
    logger.info("Analyzing posterior distributions...")

    # Plot parameter posterior distributions
    az.plot_posterior(idata, var_names=["global_σ", "σ_obs"])
    plt.tight_layout()
    plt.savefig(output_dir / "posterior_distributions.png")
    plt.close()

    # Forest plot for group-level effects
    group_vars = [var for var in idata.posterior.variables if var.endswith("_β")]
    az.plot_forest(idata, var_names=group_vars)
    plt.tight_layout()
    plt.savefig(output_dir / "group_effects.png")
    plt.close()


def analyze_prediction_accuracy(
    predictions: dict, y_true: np.ndarray, output_dir: Path
):
    """Analyze prediction accuracy and calibration."""
    logger.info("Analyzing prediction accuracy...")

    # Create accuracy plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # 1. Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_true, predictions["mean"], alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    ax.set_xlabel("Actual PIR")
    ax.set_ylabel("Predicted PIR")
    ax.set_title("Actual vs Predicted PIR")

    # 2. Error Distribution
    ax = axes[0, 1]
    errors = y_true - predictions["mean"]
    sns.histplot(errors, kde=True, ax=ax)
    ax.axvline(x=0, color="r", linestyle="--")
    ax.set_xlabel("Prediction Error")
    ax.set_title("Error Distribution")

    # 3. Residuals plot
    ax = axes[1, 0]
    ax.scatter(predictions["mean"], errors, alpha=0.5)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted PIR")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")

    # 4. Interval width vs Error
    ax = axes[1, 1]
    interval_width = predictions["upper"] - predictions["lower"]
    ax.scatter(interval_width, abs(errors), alpha=0.5)
    ax.set_xlabel("Prediction Interval Width")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Uncertainty vs Error")

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_accuracy.png")
    plt.close()

    # Calculate error statistics by prediction interval width quantiles
    df = pd.DataFrame(
        {"error": errors, "abs_error": abs(errors), "interval_width": interval_width}
    )
    width_quantiles = pd.qcut(interval_width, q=5)

    # Convert intervals to string format
    df["width_group"] = pd.Series(width_quantiles).apply(
        lambda x: f"{x.left:.2f}-{x.right:.2f}"
    )

    error_by_width = (
        df.groupby("width_group")["abs_error"].agg(["mean", "std"]).sort_index()
    )

    return error_by_width


def analyze_player_effects(idata: az.InferenceData, metadata: dict, output_dir: Path):
    """Analyze player-specific effects."""
    logger.info("Analyzing player effects...")

    # Get player effects summary
    player_effects = az.summary(idata, var_names=["player_effects"])
    player_effects.index = [
        int(idx.split("[")[-1].strip("]")) for idx in player_effects.index
    ]

    # Map player IDs
    player_effects["player_id"] = player_effects.index.map(metadata["player_map"])

    # Plot player effects distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(player_effects["mean"], kde=True)
    plt.title("Distribution of Player Effects")
    plt.xlabel("Effect Size")
    plt.savefig(output_dir / "player_effects_dist.png")
    plt.close()

    return player_effects.sort_values("mean", ascending=False)


def analyze_tier_effects(idata: az.InferenceData, metadata: dict, output_dir: Path):
    """Analyze tier-specific effects."""
    logger.info("Analyzing tier effects...")

    # Get tier effects summary
    tier_effects = az.summary(idata, var_names=["tier_effects"])
    tier_effects.index = [
        int(idx.split("[")[-1].strip("]")) for idx in tier_effects.index
    ]

    # Map tier names
    tier_effects["tier"] = tier_effects.index.map(metadata["tier_map"])

    # Plot tier effects
    plt.figure(figsize=(10, 6))
    tier_effects = tier_effects.sort_values("mean")
    plt.errorbar(
        tier_effects["mean"], range(len(tier_effects)), xerr=tier_effects["sd"], fmt="o"
    )
    plt.yticks(range(len(tier_effects)), tier_effects["tier"])
    plt.xlabel("Effect Size")
    plt.title("Tier Effects with 95% CI")
    plt.tight_layout()
    plt.savefig(output_dir / "tier_effects.png")
    plt.close()

    return tier_effects


def analyze_feature_groups(idata: az.InferenceData, metadata: dict, output_dir: Path):
    """Analyze feature group effects."""
    logger.info("Analyzing feature group effects...")

    group_effects = {}
    for group in metadata["features"].keys():
        effects = az.summary(idata, var_names=[f"{group}_β"])
        effects.index = metadata["features"][group]
        group_effects[group] = effects

        # Plot group effects
        plt.figure(figsize=(10, 6))
        effects = effects.sort_values("mean")
        plt.errorbar(effects["mean"], range(len(effects)), xerr=effects["sd"], fmt="o")
        plt.yticks(range(len(effects)), effects.index)
        plt.xlabel("Effect Size")
        plt.title(f"{group.title()} Feature Effects")
        plt.tight_layout()
        plt.savefig(output_dir / f"{group}_effects.png")
        plt.close()

    return group_effects


def main():
    # Load parameters
    params = load_params()

    # Set up paths
    data_dir = Path(params.get("data_dir", "euroleague_data"))
    model_dir = Path(params.get("model_dir", "models_pgm"))
    analysis_dir = Path(params.get("analysis_dir", "analysis_pgm"))
    analysis_dir.mkdir(exist_ok=True)

    # Load artifacts
    artifacts = load_model_artifacts(model_dir)

    # Load data
    # Load original data for true values
    df = pd.read_parquet(data_dir / "features.parquet")

    # Select features
    features = {
        "performance": ["PIR", "minutes_played", "Points"],
        "efficiency": ["fg_percentage", "ft_percentage", "ast_to_turnover"],
        "momentum": ["pir_ma3", "points_ma3", "minutes_ma3"],
        "context": ["is_starter", "is_home"],
    }
    all_features = [f for group in features.values() for f in group]

    # Remove rows with missing target or features first
    # mask = ~df["next_PIR"].isna() & df[all_features].notna().all(axis=1)
    mask = df[all_features].notna().all(axis=1)

    df = df[mask].copy()
    metadata = artifacts["metadata"]
    scalers = artifacts["scalers"]
    # Get indices
    tier_indices = pd.Categorical(df["player_tier"]).codes
    player_indices = pd.Categorical(df["Player_ID"]).codes
    idata = artifacts["inference_data"]

    X_dict = {}
    for group, features in metadata["features"].items():
        X_dict[group] = scalers[group].transform(df[features])

    # Reconstruct predictions
    idata = reconstruct_predictions(
        idata, X_dict, metadata, tier_indices, player_indices
    )

    artifacts["inference_data"] = idata

    # Analyze posterior distributions
    analyze_posterior_distributions(artifacts["inference_data"], analysis_dir)

    # Get predictions from inference data
    post_pred = artifacts["inference_data"].posterior["y_model"]

    total_effects = post_pred.sum(
        dim=["performance_β_dim_0", "efficiency_β_dim_0", "momentum_β_dim_0"]
    )

    # Now calculate statistics across chains and draws
    predictions = {
        "mean": total_effects.mean(dim=["chain", "draw"]).values,
        "std": total_effects.std(dim=["chain", "draw"]).values,
        "lower": total_effects.quantile(0.025, dim=["chain", "draw"]).values,
        "upper": total_effects.quantile(0.975, dim=["chain", "draw"]).values,
    }

    predictions = {
        "mean": predictions["mean"].sum(axis=1),  # Sum across features
        "std": np.sqrt(np.sum(predictions["std"] ** 2, axis=1)),  # Combine stds
        "lower": predictions["lower"].sum(axis=1),
        "upper": predictions["upper"].sum(axis=1),
    }

    predictions["mean"].shape

    # Analyze prediction accuracy
    error_by_width = analyze_prediction_accuracy(
        predictions, df["PIR"].values, analysis_dir
    )

    # Analyze player effects
    player_effects = analyze_player_effects(
        artifacts["inference_data"], artifacts["metadata"], analysis_dir
    )

    # Analyze tier effects
    tier_effects = analyze_tier_effects(
        artifacts["inference_data"], artifacts["metadata"], analysis_dir
    )

    # Analyze feature groups
    group_effects = analyze_feature_groups(
        artifacts["inference_data"], artifacts["metadata"], analysis_dir
    )

    group_effects["performance"]

    # Save detailed analysis results
    analysis_results = {
        "error_by_uncertainty": error_by_width.to_dict(),
        "top_players": player_effects.head(10).to_dict(),
        "tier_effects": tier_effects.to_dict(),
        "feature_effects": {
            group: effects.to_dict() for group, effects in group_effects.items()
        },
    }

    analysis_results["error_by_uncertainty"]

    with open(analysis_dir / "detailed_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

    # Print key insights
    logger.info("\nKey Insights:")
    logger.info("\nTop 5 Most Influential Players:")
    for idx, row in player_effects.head().iterrows():
        logger.info(f"{row['player_id']}: {row['mean']:.3f} ± {row['sd']:.3f}")

    logger.info("\nTier Effects:")
    for idx, row in tier_effects.iterrows():
        logger.info(f"{row['tier']}: {row['mean']:.3f} ± {row['sd']:.3f}")

    # In the main function, change this part:
    logger.info("\nUncertainty Analysis:")
    for idx, row in error_by_width.iterrows():
        # Parse the interval string back into bounds
        left, right = map(float, idx.split("-"))
        logger.info(
            f"Interval width {left:.2f}-{right:.2f}: Mean abs error = {row['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
