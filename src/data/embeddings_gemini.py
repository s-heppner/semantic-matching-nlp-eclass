import pandas as pd
from google import genai
from google.genai.types import EmbedContentConfig
from src.utils.io import save_json
from src.utils.logger import LoggerFactory

# Setup logger
logger = LoggerFactory.get_logger(__name__)

# Create Gemini client (Vertex AI) and request embeddings
def embed_eclass_file(csv_path: str, save_path: str) -> None:
    logger.info(f"Starting embedding for {csv_path}")

    client = genai.Client(
        vertexai=True, project="PLACEHOLDER", location="global" # Adapt manually
    )

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

    for idx, row in database.iterrows():
        id = row["id"]
        preferred_name = row["preferred-name"]
        description = row["definition"]

        # Skip empty descriptions
        if (
                pd.isna(description)
                or not isinstance(description, str)
                or description.strip() in ["-", "-no definition"]
        ):
            logger.warning(f"Skipping row {idx}: invalid description.")
            continue

        # Request one embedding, no batch mode
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=description,
                config=EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=3072
                )
            )
            embedding = result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding row {idx}: {e}")
            continue

        # Store result
        entry = {
            "id": id,
            "preferred-name": preferred_name,
            "description": description,
            "embedding": embedding
        }
        embedded_entries.append(entry)

        if len(embedded_entries) % 10 == 0:
            logger.info(f"Processed {len(embedded_entries)} entries.")

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

    csv_path = f"../../data/interim/eclass-{segment}.csv" # Adapt manually
    save_path = f"../../data/embeddings/eclass-{segment}-embeddings-gemini.json" # Adapt manually
    embed_eclass_file(csv_path, save_path)