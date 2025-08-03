"""
This module can calculate the matching scores using cosine similarity
"""
import json
import math
import csv
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm  # Nicer progress bars

import numpy as np

def compute_similarities(embeddings_dir: str, output_csv: str) -> None:
    file_paths: List[Path] = list(Path(embeddings_dir).glob("*.json"))

    # We begin by creating a list of all entries
    entries: List[Dict[str, object]] = []
    for path in tqdm(file_paths, desc="Indexing"):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
            for entry in data:
                entries.append({
                    "id": entry["id"],
                    "vector_norm": entry["vector_norm"],
                    "embedding": entry["embedding"]
                })

    # Now we do the actual calculation.
    with open(output_csv, "w", newline='', encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["id1", "id2", "cosine_similarity"])

        for i in tqdm(range(len(entries)), desc="Computing"):
            id1 = entries[i]["id"]
            vec1 = entries[i]["embedding"]
            norm1 = entries[i]["vector_norm"]

            # Since `s_{A,B} == s_{B,A}`, we only have to calculate each pair once
            for j in range(i + 1, len(entries)):
                id2 = entries[j]["id"]
                vec2 = entries[j]["embedding"]
                norm2 = entries[j]["vector_norm"]

                sim: float = float(np.dot(vec1, vec2) / (norm1 * norm2))
                writer.writerow([id1, id2, f"{sim:.6f}"])

if __name__ == '__main__':
    compute_similarities("../../data/embeddings/qwen3", "../../data/scores/scores_qwen3.csv")
