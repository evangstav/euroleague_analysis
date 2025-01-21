"""Configuration management for feature engineering."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    data_dir: Path
    output_dir: Path
    seasons: List[str]
    transformers: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FeatureConfig":
        """Create config from dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            FeatureConfig instance
        """
        return cls(
            data_dir=Path(config_dict["data_dir"]),
            output_dir=Path(config_dict["output_dir"]),
            seasons=config_dict["seasons"],
            transformers=config_dict.get("transformers"),
        )


def get_config(params_file: str = "params.yaml") -> FeatureConfig:
    """Load configuration from params file.

    Args:
        params_file: Path to params.yaml file

    Returns:
        FeatureConfig instance

    Raises:
        FileNotFoundError: If params file not found
        ValueError: If required configuration missing
    """
    # Check params file exists
    if not os.path.exists(params_file):
        raise FileNotFoundError(
            f"Configuration file not found: {params_file}\n"
            "Please ensure params.yaml exists in the project root."
        )

    # Load configuration
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Extract feature engineering section
    if "feature_engineering" not in params:
        raise ValueError(
            "Missing feature_engineering section in params.yaml\n"
            "Please ensure params.yaml contains feature_engineering configuration."
        )

    config_dict = params["feature_engineering"]

    # Validate required fields
    required_fields = ["data_dir", "output_dir", "seasons"]
    missing = [field for field in required_fields if field not in config_dict]
    if missing:
        raise ValueError(
            f"Missing required configuration fields: {missing}\n"
            "Please ensure all required fields are specified in params.yaml."
        )

    return FeatureConfig.from_dict(config_dict)
