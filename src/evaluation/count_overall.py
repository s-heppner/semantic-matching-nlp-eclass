import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

exceptions = []
versions = [v for v in list(range(13, 52)) + [90] if v not in exceptions]

version_labels = []
non_semantic_ratios = []
non_semantic_counts = []
total_defs = []

for version in versions:
    csv_path = f"../../data/interim/eclass-{version}.csv"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
        continue

    if "definition" not in df.columns:
        logger.warning(f"No 'definition' column in {csv_path}, skipping.")
        continue

    total = len(df)
    if total == 0:
        logger.warning(f"Version {version} has 0 entries, skipping.")
        continue

    bad_count = 0

    for x in df["definition"]:
        if pd.isna(x):
            bad_count += 1
        elif not isinstance(x, str):
            bad_count += 1
        elif x == "-":
            bad_count += 1
        elif x == "-no definition":
            bad_count += 1
        elif x.startswith("Sub-group (4th level)") or x.startswith("sub-group (4th level)"):
            bad_count += 1

    ratio = (bad_count / total) * 100 if total > 0 else 0

    version_labels.append(str(version))
    non_semantic_ratios.append(ratio)
    non_semantic_counts.append(bad_count)
    total_defs.append(total)

    logger.info(f"Version {version}: {bad_count}/{total} non-semantic definitions ({ratio:.2f}%).")

# Plot one bar per segment
plt.figure(figsize=(16, 8))
plt.bar(version_labels, non_semantic_ratios, color="#4c72b0")

# Add text above each bar
for i, (count, total, ratio) in enumerate(zip(non_semantic_counts, total_defs, non_semantic_ratios)):
    if ratio > 0:
        label_text = f"{count}\n—\n{total}"
        y_pos = ratio + 2
        plt.text(
            i, y_pos,
            label_text,
            ha="center", va="bottom", fontsize="x-small"
        )

# Dynamically extend y-axis
max_ratio = max(non_semantic_ratios)
plt.ylim(0, max_ratio * 1.2 if max_ratio > 0 else 10)

plt.xlabel("ECLASS Segments")
plt.ylabel("Non-Semantic Definition Ratio (%)")
plt.title("Non-Semantic Definitions per ECLASS Segment")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("../../visualisation/non_semantic_definitions_ratio.png")
logger.info("Plot saved to ../../visualisation/non_semantic_definitions_ratio.png")

plt.show()