"""Module to embed ECLASS definitions using the BAAI/bge-large-en-v1.5 transformer model."""

from embedder import EclassEmbedder

if __name__ == "__main__":
    # Settings
    apply_filters = True  # Enable filtering of non-semantic definitions
    exceptions = []  # Exclude specific segments

    # Setup
    embedder = EclassEmbedder(model_name="BAAI/bge-large-en-v1.5")
    save_dir = "filtered" if apply_filters else "unfiltered"
    segments = list(range(13, 52)) + [90]

    # Run for each segment
    for segment in segments:

        if segment in exceptions:
            embedder.logger.warning(f"Skipping segment {segment}.")
            continue
        input_path = f"../../data/extracted/eclass-{segment}.csv"
        output_path = f"../../data/embedded/{save_dir}/eclass-{segment}-embeddings-bge.json"
        embedder.embed_eclass(input_path, output_path, apply_filters=apply_filters)
