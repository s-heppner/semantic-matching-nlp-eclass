"""Module to embed ECLASS definitions using the gemini-embedding-001 transformer model."""

import logging
import numpy as np
import pandas as pd
from google import genai
from google.genai.types import EmbedContentConfig
from src.embedding.filter import (
    filter_definitions_missing,
    filter_definitions_missing_suffix,
    filter_definitions_structural,
)
from src.utils.io import save_json
from src.utils.logger import LoggerFactory


def embed_eclass(
        input_path: str,
        output_path: str,
        logger: logging.Logger,
        apply_filters: bool = True
) -> None:
    """Loads ECLASS data, filters definitions, computes embeddings with Gemini, and saves the results as JSON."""

    logger.info(f"Starting embedder ...")
    logger.info(f"Filters enabled: {apply_filters}")

    # Load ECLASS classes from CSV file
    try:
        database = pd.read_csv(input_path, sep=",")
        logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
    except Exception as e:
        logger.error(f"Failed to read file: {input_path}, Error: {e}")
        return

    # Ensure required columns exist
    required_columns = ["definition", "preferred-name", "id"]
    for col in required_columns:
        if col not in database.columns:
            logger.error(f"Missing required column: {col}")
            return

    # Filter definitions if enabled
    valid_rows = []
    for idx, row in database.iterrows():
        definition = row["definition"]

        if apply_filters:
            if (
                    definition in filter_definitions_missing
                    or any(definition.endswith(suffix) for suffix in filter_definitions_missing_suffix)
                    or definition in filter_definitions_structural
            ):
                logger.warning(f"Skipping row {idx}: non-semantic definition '{definition}'")
                continue

        valid_rows.append(row)
    logger.info(f"Total valid rows for embedding: {len(valid_rows)}")

    # Compute embeddings, no batch mode
    definitions = [row["definition"] for row in valid_rows]
    embeddings = []
    logger.info("Computing embeddings ...")
    for definition in definitions:
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=definition,
                config=EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=3072
                )
            )
            embeddings.append(result.embeddings[0].values)
        except Exception as e:
            logger.error(f"Error while embedding: {e}")
            continue
        if len(embeddings) % 10 == 0:
            logger.info(f"Processed {len(embeddings)} entries.")

    # Save results
    embedded_entries = [
        {
            "id": row["id"],
            "preferred-name": row["preferred-name"],
            "definition": row["definition"],
            "vector-norm": float(np.linalg.norm(embedding)),
            "embedding": embedding,
        }
        for row, embedding in zip(valid_rows, embeddings)
    ]
    logger.info(f"Embedding computation finished for {input_path}. Total processed: {len(embedded_entries)}")
    save_json(embedded_entries, output_path)
    logger.info(f"Saved embeddings to: {output_path}")


if __name__ == "__main__":
    # Settings
    apply_filters = True  # Enable filtering of non-semantic definitions
    run_per_segment = True  # Compute embeddings per segment or in one run
    exceptions = []  # Exclude specific segments if run per segment
    project = "PLACEHOLDER"  # Project name from Google Cloud

    # Setup
    logger = LoggerFactory.get_logger(__name__)
    logger.info("Initialising embedder ...")
    save_dir = "filtered" if apply_filters else "unfiltered"

    # Connect Gemini client via Vertex Ai
    client = genai.Client(vertexai=True, project=project, location="global")
    logger.info(f"Client loaded.")

    if run_per_segment:
        # Run for each segment
        segments = list(range(13, 52)) + [90]
        for segment in segments:
            if segment in exceptions:
                logger.warning(f"Skipping segment {segment}.")
                continue
            input_path = f"../../data/extracted/eclass-{segment}.csv"
            output_path = f"../../data/embedded/{save_dir}/eclass-{segment}-embeddings-gemini.json"
            embed_eclass(input_path, output_path, logger, apply_filters=apply_filters)
    else:
        # Run for combined segments
        input_path = f"../../data/extracted/eclass-all.csv"
        output_path = f"../../data/embedded/{save_dir}/eclass-all-embeddings-gemini.json"
        embed_eclass(input_path, output_path, logger, apply_filters=apply_filters)
