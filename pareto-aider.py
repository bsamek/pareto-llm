import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create the dataset
data = {
    "model": [
        "o3 (high) + gpt-4.1",
        "o3 (high)",
        "Gemini 2.5 Pro Preview 05-06",
        "o4-mini (high)",
        "claude-3-7-sonnet-20250219 (32k)",
        "DeepSeek R1 + claude-3-5-sonnet",
        "claude-3-7-sonnet (no thinking)",
        "DeepSeek R1",
        "DeepSeek V3 (0324)",
        "Grok 3 Beta",
        "gpt-4.1",
        "Grok 3 Mini Beta (high)",
        "gemini-2.5-flash-preview-04-17",
        "chatgpt-4o-latest (2025-03-29)",
        "Grok 3 Mini Beta (low)",
        "gpt-4.1-mini",
    ],
    "accuracy": [
        82.7,
        79.6,
        76.9,
        72.0,
        64.9,
        64.0,
        60.4,
        56.9,
        55.1,
        53.3,
        52.4,
        49.3,
        47.1,
        45.3,
        34.7,
        32.4,
    ],
    "cost": [
        69.29,
        111.03,
        37.41,
        19.64,
        36.83,
        13.29,
        17.72,
        5.42,
        1.12,
        11.03,
        9.86,
        0.73,
        1.85,
        19.74,
        0.79,
        1.99,
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
