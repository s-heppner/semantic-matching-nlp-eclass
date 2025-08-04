import json
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import plotly.express as px
import os

from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

exceptions = []
segments = list(range(13, 52)) + [90]
segments = [s for s in segments if s not in exceptions]

for segment in segments:
    logger.info(f"Processing segment {segment}")

    json_path = f"../../data/embeddings/original/eclass-{segment}-embeddings-qwen3.json"
    save_path = f"../../visualisation/eclass-{segment}-embeddings-qwen3.html"

    if not os.path.exists(json_path):
        logger.warning(f"Skipping segment {segment}: JSON file not found at {json_path}")
        continue

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    embeddings = np.array(df["embedding"].tolist())

    # DBSCAN clustering
    embeddings = normalize(embeddings, norm='l2')
    dbscan = DBSCAN(eps=0.05, min_samples=5, metric="cosine")
    labels = dbscan.fit_predict(embeddings)
    df["cluster"] = labels

    unique_clusters = sorted(set(labels))
    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Skip noise
        examples = df[df["cluster"] == cluster]["preferred-name"].head(5).tolist()
        logger.info(f"Segment {segment} - Cluster {cluster} example preferred names: {examples}")

    # UMAP reduction to 2D
    reducer = umap.UMAP(metric="cosine", random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=df["cluster"].astype(str),
        hover_data=["preferred-name", "description"],
        title=f"Preferred Names Embedding Map (UMAP, DBSCAN Clusters) - Segment {segment}"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7), selector=dict(mode='markers'))
    fig.update_layout(
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        showlegend=True
    )

    # Save HTML
    fig.write_html(save_path)
    logger.info(f"DBSCAN plot saved to {save_path}")