import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

exceptions = []
versions = [v for v in list(range(13, 52)) + [90] if v not in exceptions]

# Count placeholders
placeholder_counts = defaultdict(lambda: defaultdict(int))
segment_totals = {}
version_labels = []

for version in versions:
    csv_path = f"../../data/extracted/eclass-{version}.csv"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
        continue

    if "definition" not in df.columns:
        logger.warning(f"No 'definition' column in {csv_path}, skipping.")
        continue

    total = len(df)
    segment_totals[version] = total

    placeholders = df["definition"].dropna().loc[
        lambda x: x.str.startswith("Sub-group (4th level)") | x.str.startswith("sub-group (4th level)")
    ]

    for placeholder_text, count in placeholders.value_counts().items():
        placeholder_counts[placeholder_text][version] = count

    version_labels.append(str(version))
    logger.info(f"Version {version}: {len(placeholders)} placeholder definitions found out of {total} total entries.")

# Create DataFrame
df_counts = pd.DataFrame(placeholder_counts).fillna(0).astype(int)
df_counts = df_counts.T

# Calculate per-segment placeholder sums for annotation
segment_placeholder_sums = {v: df_counts[v].sum() for v in versions}

# Convert to ratio
for version in versions:
    total = segment_totals.get(version, 1)
    df_counts[version] = df_counts[version] / total * 100  # in %

df_counts["total"] = df_counts.sum(axis=1)
df_counts = df_counts.sort_values("total", ascending=False)
df_counts = df_counts.drop(columns=["total"])

# Color palette
cmap = plt.get_cmap("tab20")
colors = [cmap(i % cmap.N) for i in range(len(df_counts))]

plt.figure(figsize=(16, 8))

bottom = [0] * len(versions)
for idx, (placeholder_text, row) in enumerate(df_counts.iterrows()):
    ratios = [row.get(v, 0) for v in versions]
    plt.bar(
        [str(v) for v in versions],
        ratios,
        bottom=bottom,
        label=placeholder_text,
        color=colors[idx]
    )
    bottom = [b + r for b, r in zip(bottom, ratios)]

# Add annotations above bars
for i, v in enumerate(versions):
    placeholder_sum = segment_placeholder_sums[v]
    total_defs = segment_totals[v]
    ratio_sum = bottom[i]

    # Only add if the bar is non-zero
    if ratio_sum > 0:
        label_text = f"{placeholder_sum}\n—\n{total_defs}"
        y_pos = ratio_sum + 2  # offset above bar
        plt.text(
            i, y_pos,
            label_text,
            ha="center", va="bottom", fontsize="x-small"
        )

# Dynamically extend y-axis to avoid text cutoff
max_ratio = max(bottom)
plt.ylim(0, max_ratio * 1.2)

plt.xlabel("ECLASS Segments")
plt.ylabel("Placeholder Definition Ratio (%)")
plt.title("Stacked Causes of Placeholder Definitions per ECLASS Segment")
plt.xticks(rotation=45)

# Legend below plot
plt.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig("../../visualisation/placeholder_definitions_ratio.png", bbox_inches="tight")
logger.info("Plot saved to ../../visualisation/placeholder_definitions_ratio.png")

plt.show()