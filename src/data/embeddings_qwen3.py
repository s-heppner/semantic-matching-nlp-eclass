import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils.io import save_json
from src.utils.logger import LoggerFactory
import torch

# Setup logger
logger = LoggerFactory.get_logger(__name__)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load model
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"torch_dtype": torch.float16}, # 16-bit floating point, uses less memory, adapt manually
    device=device,
    tokenizer_kwargs={"padding_side": "left"},
)

# Compute embeddings
def embed_eclass_file(csv_path: str, save_path: str) -> None:
    logger.info(f"Starting embedding for {csv_path}")

    # Load data
    database = pd.read_csv(csv_path, sep=",")
    logger.info(f"Database loaded from {csv_path} with {len(database)} rows.")

    required_columns = ["definition", "preferred-name", "id"]
    for col in required_columns:
        if col not in database.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")

    # Set up embedding computation
    embedded_entries = []
    logger.info("Starting embedding computation...")

    valid_rows = []

    for idx, row in database.iterrows():
        description = row["definition"]

        # Skip empty descriptions
        if (
            pd.isna(description)
            or not isinstance(description, str)
            or description.strip() in ["-", "-no definition"]
        ):
            logger.warning(f"Skipping row {idx}: invalid description.")
            continue
        valid_rows.append(row)

    # Compute 32 embeddings, batch mode
    descriptions = [row["definition"] for row in valid_rows]
    embeddings = model.encode(descriptions, batch_size=32, show_progress_bar=True)

    # Store result
    for row, embedding in zip(valid_rows, embeddings):
        entry = {
            "id": row["id"],
            "preferred-name": row["preferred-name"],
            "description": row["definition"],
            "embedding": embedding.tolist()
        }
        embedded_entries.append(entry)

    logger.info(f"Embedding computation finished for {csv_path}. Total processed: {len(embedded_entries)}")

    # Save results
    save_json(embedded_entries, save_path)
    logger.info(f"Embeddings saved to {save_path}")

# Execute for all segments
exceptions = [] # Adapt manually
for segment in list(range(13, 52)) + [90]:
    if segment in exceptions:
        logger.info(f"Skipping segment {segment}.")
        continue

    csv_path = f"../../data/extracted/eclass-{segment}.csv" # Adapt manually
    save_path = f"../../data/embeddings/original/eclass-{segment}-embeddings-qwen3.json" # Adapt manually
    embed_eclass_file(csv_path, save_path)
