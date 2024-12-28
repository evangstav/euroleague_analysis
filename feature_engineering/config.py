"""Configuration module for feature engineering."""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import logging
import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""

    season: int
    data_dir: Path
    db_path: str
    output_dir: Path

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "FeatureConfig":
        """Create config from params dictionary"""
        data_dir = Path(params.get("data_dir", "euroleague_data"))
        return cls(
            season=params.get("season", 2023),
            data_dir=data_dir,
            db_path=str(data_dir / "features.duckdb"),
            output_dir=data_dir,
        )


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f).get("features", {})
    return params
