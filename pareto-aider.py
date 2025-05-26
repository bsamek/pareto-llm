import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create the dataset
data = {
    "model": [
        "o3 (high) + gpt-4.1",
        "o3 (high)",
        "Gemini 2.5 Pro Preview 05-06",
        "claude-opus-4-20250514 (32k thinking)",
        "o4-mini (high)",
        "claude-opus-4-20250514 (no think)",
        "claude-3-7-sonnet-20250219 (32k thinking tokens)",
        "DeepSeek R1 + claude-3-5-sonnet-20241022",
        "o1-2024-12-17 (high)",
        "claude-sonnet-4-20250514 (32k thinking)",
        "claude-3-7-sonnet-20250219 (no thinking)",
        "o3-mini (high)",
        "DeepSeek R1",
        "claude-sonnet-4-20250514 (no thinking)",
        "DeepSeek V3 (0324)",
        "o3-mini (medium)",
        "Grok 3 Beta",
        "gpt-4.1",
        "claude-3-5-sonnet-20241022",
        "Grok 3 Mini Beta (high)",
        "DeepSeek Chat V3 (prev)",
        "gemini-2.5-flash-preview-04-17 (default)",
        "chatgpt-4o-latest (2025-03-29)",
        "gpt-4.5-preview",
        "Qwen3 32B",
        "Grok 3 Mini Beta (low)",
        "o1-mini-2024-09-12",
        "gpt-4.1-mini",
        "claude-3-5-haiku-20241022",
        "chatgpt-4o-latest (2025-02-15)",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "DeepSeek Chat V2.5",
        "Codestral 25.01",
        "gpt-4.1-nano",
        "gpt-4o-mini-2024-07-18",
    ],
    "accuracy": [
        82.7,
        79.6,
        76.9,
        72.0,
        72.0,
        70.7,
        64.9,
        64.0,
        61.7,
        61.3,
        60.4,
        60.4,
        56.9,
        56.4,
        55.1,
        53.8,
        53.3,
        52.4,
        51.6,
        49.3,
        48.4,
        47.1,
        45.3,
        44.9,
        40.0,
        34.7,
        32.9,
        32.4,
        28.0,
        27.1,
        23.1,
        18.2,
        17.8,
        11.1,
        8.9,
        3.6,
    ],
    "cost": [
        69.29,
        111.03,
        37.41,
        65.75,
        19.64,
        68.63,
        36.83,
        13.29,
        186.5,
        26.58,
        17.72,
        18.16,
        5.42,
        15.82,
        1.12,
        8.86,
        11.03,
        9.86,
        14.41,
        0.73,
        0.34,
        1.85,
        19.74,
        183.18,
        0.76,
        0.79,
        18.58,
        1.99,
        6.06,
        14.37,
        7.03,
        6.74,
        0.51,
        1.98,
        0.43,
        0.32,
    ],
}
df = pd.DataFrame(data)

# Identify Pareto frontier
pareto_models = []
df_sorted = df.sort_values("cost")

for idx, row in df_sorted.iterrows():
    dominated = False
    for idx2, row2 in df_sorted.iterrows():
        if row2["cost"] < row["cost"] and row2["accuracy"] >= row["accuracy"]:
            dominated = True
            break
    if not dominated:
        pareto_models.append(row["model"])

# Create the plot
plt.figure(figsize=(12, 8))

# Plot all models with consistent coloring
for idx, row in df.iterrows():
    color = "blue"  # Use blue for all models
    marker = (
        "s" if row["model"] in pareto_models else "o"
    )  # Square for frontier, circle for others
    size = 120 if row["model"] in pareto_models else 60
    edgecolor = "black" if row["model"] in pareto_models else "none"
    linewidth = 2 if row["model"] in pareto_models else 0

    plt.scatter(
        row["cost"],
        row["accuracy"],
        color=color,
        marker=marker,
        s=size,
        zorder=5 if row["model"] in pareto_models else 3,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=0.8,
    )

    # Add labels with appropriate formatting based on frontier status
    if row["model"] in pareto_models:
        plt.text(
            row["cost"],
            row["accuracy"] + 0.5,
            row["model"],
            fontsize=8,
            ha="center",
            rotation=45,
            fontweight="bold",
        )
    else:
        plt.text(
            row["cost"],
            row["accuracy"] + 0.5,
            row["model"],
            fontsize=6,
            ha="center",
            alpha=0.7,
            rotation=45,
        )

# Connect Pareto frontier points
pareto_df = df[df["model"].isin(pareto_models)].sort_values("cost")
plt.plot(
    pareto_df["cost"], pareto_df["accuracy"], "k-", linewidth=2, alpha=0.5, zorder=4
)

plt.xscale("log")
plt.xlabel("Cost ($)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("LLM Pareto Frontier: Cost vs Accuracy", fontsize=16, fontweight="bold")
plt.grid(True, alpha=0.3)

# Add legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="blue",
        markeredgecolor="black",
        markersize=10,
        label="Pareto Frontier",
        linewidth=1,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=8,
        label="Other Models",
    ),
]
plt.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.show()
