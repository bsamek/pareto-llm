import pandas as pd
import plotly.graph_objects as go

# Create the dataset
data = {
    "model": [
        "o3 [CoT]",
        "claude-4-opus [CoT]",
        "gemini-2-5-pro [CoT]",
        "arcee-ai/maestro [CoT]",
        "qwen-3-235b-a22b-rope [CoT]",
        "o1 [CoT]",
        "o1-pro [CoT]",
        "claude-4-sonnet [CoT]",
        "o4-mini [CoT]",
        "claude-3-7-sonnet [CoT]",
        "qwen-qwq-32b [CoT]",
        "deepseek-r1 [CoT]",
        "o3-mini [CoT]",
        "grok-3-mini [CoT-high]",
        "qwen-3-235b-a22b [CoT]",
        "grok-3-mini [CoT-low]",
        "claude-4-opus [no-think]",
        "perplexity/sonar-pro [CoT]",
        "chatgpt-4o",
        "gpt-4-1",
        "deepseekr1-distil-llama[CoT]",
        "claude-4-sonnet [no-think]",
        "deepseek-chat-v3",
        "qwen-3-32b [CoT]",
        "thedrummer/anubis-pro-105b-v1",
        "llama-4-maverick",
        "gpt-4-1-mini",
        "grok-3",
        "mistral-medium",
        "qwen-3-235b-a22b [no-think]",
        "gpt-4o",
        "gemini-2-5-flash [no-think]",
        "mistral-large",
        "claude-3-5-sonnet",
        "mistral-small",
        "aion-labs/aion-1.0-mini",
        "claude-3-opus",
        "llama-3-405b",
        "minimax/minimax-01",
        "arcee-ai/virtuoso-large",
        "gpt-4-turbo",
        "gemini-flash",
        "qwen-3-32b [no-think]",
        "gpt-4",
        "gpt-4o-mini",
        "claude-3-opus",
        "gemini-1-5-pro",
        "llama-4-scout",
        "llama-3-70b",
        "qwen-2-5-vl-72b",
        "claude-3-haiku",
        "qwen-vl-max",
        "nova-pro",
        "llama-3-3b",
        "mistral-nemo",
        "gemma-3-27b",
        "nova-lite",
        "cohere/command-r7b-12-2024",
        "gemma2-9b-it",
        "liquid/lfm-40b",
    ],
    "accuracy": [
        78.03,
        77.46,
        74.03,
        72.92,
        72.12,
        71.73,
        70.92,
        68.69,
        67.79,
        65.15,
        64.60,
        64.00,
        63.37,
        62.58,
        61.03,
        60.76,
        60.21,
        55.21,
        54.80,
        53.76,
        52.62,
        52.58,
        51.95,
        49.83,
        48.96,
        48.93,
        48.80,
        48.40,
        47.20,
        43.00,
        42.60,
        41.88,
        40.53,
        38.89,
        37.99,
        37.20,
        37.09,
        37.07,
        36.40,
        35.42,
        34.27,
        34.10,
        33.93,
        33.44,
        33.38,
        31.23,
        31.20,
        30.80,
        29.21,
        28.36,
        26.44,
        25.31,
        23.63,
        22.79,
        22.40,
        21.79,
        21.04,
        19.82,
        19.27,
        17.71,
    ],
    "cost": [
        2.08,
        2.84,
        0.19,
        0.25,
        0.08,
        5.11,
        47.64,
        0.62,
        0.32,
        1.81,
        0.87,
        0.97,
        0.42,
        0.28,
        0.06,
        0.02,
        1.06,
        0.11,
        0.57,
        0.18,
        0.33,
        0.23,
        0.24,
        0.03,
        0.02,
        0.03,
        0.05,
        0.70,
        0.05,
        0.02,
        0.22,
        0.02,
        0.10,
        0.23,
        0.00,
        0.08,
        1.29,
        0.27,
        0.02,
        0.02,
        1.09,
        0.01,
        0.02,
        2.22,
        0.02,
        1.06,
        0.38,
        0.02,
        0.06,
        0.01,
        0.09,
        0.06,
        0.13,
        0.01,
        0.00,
        0.01,
        0.01,
        0.00,
        None,
        0.00,
    ],
}

df = pd.DataFrame(data)

# Filter out models with no cost data or zero cost
df = df[df["cost"].notna()]
df = df[df["cost"] > 0].copy()

# For each cost, find the model with the highest accuracy
df_best_at_cost = df.loc[df.groupby("cost")["accuracy"].idxmax()]

# Identify Pareto frontier from these best models
df_sorted = df_best_at_cost.sort_values(by=["cost", "accuracy"], ascending=[True, True])
pareto_models = []
last_best_accuracy = -1

for index, row in df_sorted.iterrows():
    if row["accuracy"] > last_best_accuracy:
        pareto_models.append(row["model"])
        last_best_accuracy = row["accuracy"]

# Create the plot
fig = go.Figure()

# Add all models
fig.add_trace(
    go.Scatter(
        x=df["cost"],
        y=df["accuracy"],
        mode="markers",
        marker=dict(
            color="blue",  # Use blue for all models for consistency
            size=8,
            symbol=["square" if m in pareto_models else "circle" for m in df["model"]],
            line=dict(
                color="black",
                width=[2 if m in pareto_models else 0 for m in df["model"]],
            ),
        ),
        hovertext=df["model"],
        hoverinfo="text",
        name="Models",
    )
)

# Connect Pareto frontier points
pareto_df = df[df["model"].isin(pareto_models)].sort_values("cost")
fig.add_trace(
    go.Scatter(
        x=pareto_df["cost"],
        y=pareto_df["accuracy"],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Pareto Frontier",
    )
)

fig.update_layout(
    title="LLM Pareto Frontier: Cost vs Accuracy (Kagi)",
    xaxis_title="Cost ($)",
    yaxis_title="Accuracy (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
