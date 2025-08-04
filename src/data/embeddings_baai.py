import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from src.utils.io import save_json
from src.utils.logger import LoggerFactory

# Setup logger
logger = LoggerFactory.get_logger(__name__)
logger.info("Starting embedding script...")
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
model.to(device)
logger.info("Model loaded.")

# Load data
csv_path = "../../data/extracted/eclass-35.csv"  # Manually adapt
database = pd.read_csv(csv_path, sep=",")
logger.info(f"Database loaded from {csv_path} with {len(database)} rows.")

required_columns = ["definition", "preferred-name"]
for col in required_columns:
    if col not in database.columns:
        logger.error(f"Missing required column: {col}")
        raise ValueError(f"Missing required column: {col}")

# Set up embedding computation
embedded_entries = []
logger.info("Starting embedding computation...")

for idx, row in database.iterrows():
    id = row["id"]
    preferred_name = row["preferred-name"]
    description = row["definition"]

    # Skip empty descriptions
    if pd.isna(description) or not isinstance(description, str) or description == "-":
        logger.warning(f"Skipping row {idx}: invalid description.")
        continue

    # Compute one embedding, no batch mode
    with torch.no_grad():
        embedding = model.encode(description, convert_to_tensor=False)

    # Store result
    entry = {
        "id": id,
        "preferred-name": preferred_name,
        "description": description,
        "embedding": embedding.tolist()
    }
    embedded_entries.append(entry)

    if len(embedded_entries) % 100 == 0:
        logger.info(f"Processed {len(embedded_entries)} entries.")

logger.info(f"Embedding computation finished for {csv_path}. Total processed: {len(embedded_entries)}")

# Save results
save_path = "../../data/embeddings/original/eclass-35-embeddings-baai.json"  # Manually adapt
save_json(embedded_entries, save_path)
logger.info(f"Embeddings saved to {save_path}")
