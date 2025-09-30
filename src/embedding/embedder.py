"""Module to embed ECLASS definitions using a specified SentenceTransformer model."""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.embedding.filter import (filter_definitions_missing,
                                  filter_definitions_missing_suffix,
                                  filter_definitions_structural)
from src.utils.io import save_json
from src.utils.logger import LoggerFactory


class EclassEmbedder:
    """
    Embeds ECLASS definitions using a SentenceTransformer model.

    This class loads a specified SentenceTransformer model and applies it to compute embeddings for ECLASS definitions,
    with optional filtering.
    """

    def __init__(self, model_name: str, model_kwargs: dict = None, tokenizer_kwargs: dict = None):
        """Initialises the embedder with a specified model."""

        # Setup logger
        self.logger = LoggerFactory.get_logger(__name__)
        self.logger.info("Initialising embedder ...")

        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        # Load model
        model_kwargs = model_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            # model_name_or_path="../../models/model_name",  # Use this when downloading the model manually
            device=self.device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs
        )
        self.logger.info(f"Model '{model_name}' loaded.")

    def embed_eclass(self, input_path: str, output_path: str, apply_filters: bool = True, batch_size: int = 32) -> None:
        """Loads ECLASS data, filters definitions, computes embeddings, and saves the results as JSON."""

        self.logger.info(f"Starting embedder ...")
        self.logger.info(f"Filters enabled: {apply_filters}")

        # Load ECLASS classes from CSV file
        try:
            database = pd.read_csv(input_path, sep=",")
            self.logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
        except Exception as e:
            self.logger.error(f"Failed to read file: {input_path}, Error: {e}")
            return

        # Ensure required columns exist
        required_columns = ["definition", "preferred-name", "id"]
        for col in required_columns:
            if col not in database.columns:
                self.logger.error(f"Missing required column: {col}")
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
                    self.logger.warning(f"Skipping row {idx}: non-semantic definition '{definition}'")
                    continue

            valid_rows.append(row)
        self.logger.info(f"Total valid rows for embedding: {len(valid_rows)}")

        # Compute embeddings in batch mode
        definitions = [row["definition"] for row in valid_rows]
        self.logger.info("Computing embeddings ...")
        embeddings = self.model.encode(
            definitions,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        # Save results
        embedded_entries = [
            {
                "id": row["id"],
                "preferred-name": row["preferred-name"],
                "definition": row["definition"],
                "vector-norm": float(np.linalg.norm(embedding)),
                "embedding": embedding.tolist(),
            }
            for row, embedding in zip(valid_rows, embeddings)
        ]
        self.logger.info(f"Embedding computation finished for {input_path}. Total processed: {len(embedded_entries)}")
        save_json(embedded_entries, output_path)
        self.logger.info(f"Saved embeddings to: {output_path}")
