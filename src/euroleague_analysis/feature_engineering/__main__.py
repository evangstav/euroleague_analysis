"""Main entry point for feature engineering."""

import logging

from src.euroleague_analysis.feature_engineering.builder import FeatureBuilder
from src.euroleague_analysis.feature_engineering.config import (
    FeatureConfig,
    load_params,
)

logger = logging.getLogger(__name__)


def main():
    try:
        # Load parameters
        params = load_params()
        config = FeatureConfig.from_params(params)

        # Initialize and run feature builder
        builder = FeatureBuilder(config)

        try:
            builder.create_feature_views()
            builder.create_final_features()
            builder.save_metadata()

        finally:
            builder.close()

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
