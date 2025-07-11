import pandas as pd
import plotly.graph_objects as go
from pricing import load_pricing_data, get_model_cost

# Create the dataset combining LiveBench scores with LLM pricing data
# LiveBench Global Average scores from https://livebench.ai/ (2025-05-30)
# Pricing data loaded from llm-prices.json
#
# COST MULTIPLIERS APPLIED:
# 1. Thinking token cost adjustments (simplified to consistent 2x):
#    - All thinking models: 2x base cost vs non-thinking
#    - Applied to: Claude Opus/Sonnet Thinking, Gemini Pro Max Thinking, Gemini Flash
#
# 2. High vs Medium model cost adjustments:
#    - "High" models: 2x cost vs "Medium" models
#    - Applied to: o3 High vs o3 Medium, o4-Mini High vs o4-Mini Medium

# Load pricing data from JSON file
pricing_data = load_pricing_data()

# Model list and accuracy scores
models = [
    "o3 High",
    "o3 Medium",
    "o4-Mini High",
    "o4-Mini Medium",
    "Claude 4 Opus Thinking",
    "Claude 4 Opus",
    "Claude 4 Sonnet Thinking",
    "Claude 4 Sonnet",
    "Gemini 2.5 Pro Preview (2025-06-05 Max Thinking)",
    "Gemini 2.5 Pro Preview (2025-06-05)",
    "Gemini 2.5 Flash Preview (2025-05-20)",
    "GPT-4.1",
    "ChatGPT-4o",
    "GPT-4.1 Mini",
    "GPT-4.1 Nano",
]

accuracy_scores = [
    74.61,
    71.98,
    71.52,
    66.87,
    72.93,
    65.93,
    72.08,
    63.37,
    70.95,
    69.39,
    64.42,
    55.90,
    54.74,
    51.57,
    40.51,
]

# Calculate costs using the pricing module
costs = [get_model_cost(model, pricing_data) for model in models]

data = {
    "model": models,
    "accuracy": accuracy_scores,
    "cost": costs,
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
fig = go.Figure()

# Add all models
fig.add_trace(
    go.Scatter(
        x=df["cost"],
        y=df["accuracy"],
        mode="markers+text",
        marker=dict(
            color="blue",
            size=8,
            symbol=["square" if m in pareto_models else "circle" for m in df["model"]],
            line=dict(
                color="black",
                width=[2 if m in pareto_models else 0 for m in df["model"]],
            ),
        ),
        text=df["model"],
        textposition="top center",
        textfont=dict(size=10),
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
    title="LLM Pareto Frontier: Cost vs LiveBench Global Score",
    xaxis_title="Cost ($) - Average of Input+Output per Million Tokens",
    yaxis_title="LiveBench Global Score (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-livebench.png", width=1200, height=800)
