import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create the dataset
data = {
    "model": [
        "o3",
        "claude-4-opus",
        "gemini-2-5-pro",
        "qwen-3-235b-a22b-rope",
        "claude-4-sonnet",
        "o4-mini",
        "deepseek-r1",
        "claude-4-opus-no-think",
        "chatgpt-4o",
        "gpt-4-1",
        "deepseekr1-distil-llama",
        "claude-4-sonnet-no-think",
        "deepseek-chat-v3",
        "qwen-3-32b",
        "llama-4-maverick",
        "gpt-4-1-mini",
        "grok-3",
        "mistral-medium",
        "qwen-3-235b-a22b-no-think",
        "gpt-4o",
        "gemini-2-5-flash-no-think",
        "mistral-large",
        "mistral-small",
        "gemini-flash",
        "qwen-3-32b-no-think",
        "gpt-4o-mini",
        "llama-4-scout",
        "claude-3-haiku",
    ],
    "CoT": [
        "Y",
        "Y",
        "Y",
        "Y",
        "Y",
        "Y",
        "Y",
        "N",
        "N",
        "N",
        "Y",
        "N",
        "N",
        "Y",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
        "N",
    ],
    "accuracy": [
        78.03,
        77.46,
        74.03,
        72.12,
        68.69,
        67.79,
        64.00,
        60.21,
        54.80,
        53.76,
        52.62,
        52.58,
        51.95,
        49.83,
        48.93,
        48.80,
        48.40,
        47.20,
        43.00,
        42.60,
        41.88,
        40.53,
        37.99,
        34.10,
        33.93,
        33.38,
        30.80,
        26.44,
    ],
    "cost": [
        2.08,
        2.84,
        0.19,
        0.08,
        0.62,
        0.32,
        0.97,
        1.06,
        0.57,
        0.18,
        0.33,
        0.23,
        0.24,
        0.03,
        0.03,
        0.05,
        0.70,
        0.05,
        0.02,
        0.22,
        0.02,
        0.10,
        0.00,
        0.01,
        0.02,
        0.02,
        0.02,
        0.09,
    ],
}

df = pd.DataFrame(data)

# Filter out zero-cost models (data errors)
df = df[df["cost"] > 0]

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

# Plot all models with CoT-based coloring
for idx, row in df.iterrows():
    color = "red" if row["CoT"] == "Y" else "blue"
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
        # Offset frontier model labels to avoid overlaps
        if row["model"] == "qwen-3-235b-a22b-no-think":
            label_offset = 3.0
        elif row["model"] == "gemini-2-5-flash-no-think":
            label_offset = 3.0
        elif row["model"] == "llama-4-maverick":
            label_offset = 2.0
        elif row["model"] == "qwen-3-32b":
            label_offset = 1.0
        else:
            label_offset = 0.5
        plt.text(
            row["cost"],
            row["accuracy"] + label_offset,
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

# Connect Pareto frontier points in proper order
pareto_df = df[df["model"].isin(pareto_models)].sort_values("cost")
if len(pareto_df) > 1:
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
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label="CoT Models",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label="Non-CoT Models",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="gray",
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
        markerfacecolor="gray",
        markersize=8,
        label="Other Models",
    ),
]
plt.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.show()
