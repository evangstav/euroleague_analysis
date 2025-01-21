"""Feature engineering entry point."""

import logging
import sys
from pathlib import Path

from .config import get_config
from .pipeline import FeaturePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("feature_engineering.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Run feature engineering pipeline."""
    try:
        # Load and validate configuration
        logger.info("Loading configuration")
        config = get_config()

        logger.info(
            f"Configuration loaded:\n"
            f"  Data directory: {config.data_dir}\n"
            f"  Output directory: {config.output_dir}\n"
            f"  Seasons: {config.seasons}\n"
            f"  Transformers: {config.transformers or 'all'}"
        )

        # Initialize pipeline
        logger.info("Initializing feature pipeline")
        pipeline = FeaturePipeline(
            data_dir=config.data_dir / "raw",
            output_dir=config.output_dir,
            seasons=config.seasons,
            transformers=config.transformers,
        )

        # Run pipeline
        logger.info("Running feature pipeline")
        features_df = pipeline.run()

        # Log summary
        logger.info(
            f"Feature engineering complete:\n"
            f"  Generated features: {len(features_df.columns)}\n"
            f"  Total samples: {len(features_df)}\n"
            f"  Seasons processed: {len(config.seasons)}\n"
            f"  Output file: {config.output_dir / 'features.parquet'}"
        )

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
