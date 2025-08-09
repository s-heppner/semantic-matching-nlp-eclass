"""Script to embed ECLASS definitions using the Qwen/Qwen3-Embedding-8B transformer model."""

import torch
from embedder import EclassEmbedder

if __name__ == "__main__":
    # Settings
    apply_filters = True  # Adapt manually: enable filtering of non-semantic definitions
    exceptions = []  # Adapt manually: exclude specific segments
    torch_dtype = torch.float16  # Adapt manually: use lower precision floating point to reduce memory usage

    # Setup
    embedder = EclassEmbedder(
        model_name="Qwen/Qwen3-Embedding-8B",
        model_kwargs={"torch_dtype": torch_dtype},
        tokenizer_kwargs={"padding_side": "left"}
    )
    save_dir = "filtered" if apply_filters else "unfiltered"
    segments = list(range(13, 52)) + [90]

    # Run for each segment
    for segment in segments:
        if segment in exceptions:
            embedder.logger.warning(f"Skipping segment {segment}.")
            continue
        input_path = f"../../data/extracted/eclass-{segment}.csv"
        output_path = f"../../data/embedded/{save_dir}/eclass-{segment}-embeddings-qwen3.json"
        embedder.embed_eclass(input_path, output_path, apply_filters=apply_filters)
