import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

# Versions to exclude
exceptions = []
versions = [v for v in list(range(13, 52)) + [90] if v not in exceptions]

# Containers for stacked parts
nan_ratios = []
not_str_ratios = []
dash_ratios = []
no_def_ratios = []
version_labels = []

# To store absolute undefined and total per segment
undefined_sums = []
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

    # Counters
    nan_count = 0
    not_str_count = 0
    dash_count = 0
    no_def_count = 0

    for x in df["definition"]:
        if pd.isna(x):
            nan_count += 1
        elif not isinstance(x, str):
            not_str_count += 1
        elif x == "-":
            dash_count += 1
        elif x == "-no definition":
            no_def_count += 1

    total_undefined = nan_count + not_str_count + dash_count + no_def_count

    if total_undefined == 0:
        nan_ratios.append(0)
        not_str_ratios.append(0)
        dash_ratios.append(0)
        no_def_ratios.append(0)
    else:
        nan_ratios.append(nan_count / total)
        not_str_ratios.append(not_str_count / total)
        dash_ratios.append(dash_count / total)
        no_def_ratios.append(no_def_count / total)

    version_labels.append(str(version))
    undefined_sums.append(total_undefined)
    total_defs.append(total)

    logger.info(
        f"Version {version}: {total_undefined}/{total} undefined entries "
        f"(NaN: {nan_count}, Not str: {not_str_count}, '-': {dash_count}, '-no definition': {no_def_count})"
    )

# Convert ratios to percentages
nan_ratios_pct = [r * 100 for r in nan_ratios]
not_str_ratios_pct = [r * 100 for r in not_str_ratios]
dash_ratios_pct = [r * 100 for r in dash_ratios]
no_def_ratios_pct = [r * 100 for r in no_def_ratios]

# Plot stacked bar chart
plt.figure(figsize=(15, 7))
colors = ["#60bd68", "#f15854", "#4c72b0", "#5da5da"]

plt.bar(version_labels, nan_ratios_pct, label="NaN", color=colors[0])
plt.bar(version_labels, not_str_ratios_pct, bottom=nan_ratios_pct, label="Not String", color=colors[1])
plt.bar(
    version_labels,
    dash_ratios_pct,
    bottom=[i + j for i, j in zip(nan_ratios_pct, not_str_ratios_pct)],
    label="\"-\"",
    color=colors[2]
)
plt.bar(
    version_labels,
    no_def_ratios_pct,
    bottom=[i + j + k for i, j, k in zip(nan_ratios_pct, not_str_ratios_pct, dash_ratios_pct)],
    label="\"-no definition\"",
    color=colors[3]
)

# Add text annotations above bars
total_heights = [i + j + k + l for i, j, k, l in zip(nan_ratios_pct, not_str_ratios_pct, dash_ratios_pct, no_def_ratios_pct)]

for i, (undef_count, total_count, height) in enumerate(zip(undefined_sums, total_defs, total_heights)):
    if height > 0:
        label_text = f"{undef_count}\n—\n{total_count}"
        y_pos = height + 2  # offset above bar
        plt.text(
            i, y_pos,
            label_text,
            ha="center", va="bottom", fontsize="x-small"
        )

# Dynamically extend y-axis
max_height = max(total_heights)
plt.ylim(0, max_height * 1.2)

plt.xlabel("ECLASS Segments")
plt.ylabel("Undefined Definition Ratio (%)")
plt.title("Stacked Causes of Undefined Definitions per ECLASS Segment")
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("../../visualisation/undefined_definitions_ratio.png")
logger.info("Plot saved to ../../visualisation/undefined_definitions_ratio.png")

plt.show()