"""Module to filter out non-semantic embeddings from ECLASS embedding files."""

from src.embedding.filter import (filter_definitions_missing,
                                  filter_definitions_missing_suffix,
                                  filter_definitions_structural)
from src.utils.io import load_json, save_json
from src.utils.logger import LoggerFactory

if __name__ == "__main__":
    # Settings
    exceptions = []  # Exclude specific segments
    transformer = "gemini"  # Filter embeddings from this transformer (qwen3, bge, gemini)

    # Setup
    logger = LoggerFactory.get_logger(__name__)

    # Run for each segment
    for segment in list(range(13, 52)) + [90]:
        if segment in exceptions:
            logger.warning(f"Skipping segment {segment}.")
            continue

        # Load embeddings from JSON file
        input_path = f"../../data/embedded/unfiltered/eclass-{segment}-embeddings-{transformer}.json"
        output_path = f"../../data/embedded/filtered/eclass-{segment}-embeddings-{transformer}.json"
        try:
            database = load_json(input_path)
            logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
        except Exception as e:
            logger.error(f"Failed to read {input_path}: {e}")
            continue

        # Filter definitions
        valid_rows = []
        for idx, row in enumerate(database):
            definition = row["definition"]

            if (
                    definition in filter_definitions_missing
                    or any(definition.endswith(suffix) for suffix in filter_definitions_missing_suffix)
                    or definition in filter_definitions_structural
            ):
                logger.warning(f"Skipping row {idx}: non-semantic definition '{definition}'")
                continue

            valid_rows.append(row)

        # Save results
        save_json(valid_rows, output_path)
        logger.info(f"Segment {segment}: {len(database) - len(valid_rows)} removed, {len(valid_rows)} saved.")
