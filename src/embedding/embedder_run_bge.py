"""Module to embed ECLASS definitions using the BAAI/bge-large-en-v1.5 transformer model."""

from embedder import EclassEmbedder

if __name__ == "__main__":
    # Settings
    apply_filters = False  # Enable filtering of non-semantic definitions
    run_per_segment = False  # Compute embeddings per segment or in one run
    exceptions = []  # Exclude specific segments if run per segment

    # Setup
    embedder = EclassEmbedder(model_name="BAAI/bge-large-en-v1.5")
    save_dir = "filtered" if apply_filters else "unfiltered"

    if run_per_segment:
        # Run for each segment
        segments = list(range(13, 52)) + [90]
        for segment in segments:
            if segment in exceptions:
                embedder.logger.warning(f"Skipping segment {segment}.")
                continue
            input_path = f"../../data/extracted/eclass-{segment}.csv"
            output_path = f"../../data/embedded/{save_dir}/eclass-{segment}-embeddings-bge.json"
            embedder.embed_eclass(input_path, output_path, apply_filters=apply_filters)
    else:
        # Run for combined segments
        input_path = f"../../data/extracted/eclass-all.csv"
        output_path = f"../../data/embedded/{save_dir}/eclass-all-embeddings-bge.json"
        embedder.embed_eclass(input_path, output_path, apply_filters=apply_filters)
