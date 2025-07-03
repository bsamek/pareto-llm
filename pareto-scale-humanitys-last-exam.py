import pandas as pd
import plotly.graph_objects as go
from pricing import load_pricing_data, get_model_cost

# Create the dataset combining Scale Humanity's Last Exam scores with LLM pricing data
# Scale Humanity's Last Exam scores from https://scale.com/leaderboard/humanitys_last_exam
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
    "Gemini 2.5 Pro Preview (2025-06-05)",
    "Gemini 2.5 Flash Preview (2025-05-20)",
    "Claude 4 Opus Thinking",
    "Claude 4 Sonnet Thinking",
    "GPT-4.1",
]

accuracy_scores = [
    20.32,
    19.20,
    18.08,
    14.28,
    21.64,
    10.96,
    10.72,
    7.76,
    5.40,
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
    title="LLM Pareto Frontier: Cost vs Scale Humanity's Last Exam Score",
    xaxis_title="Cost ($) - Average of Input+Output per Million Tokens",
    yaxis_title="Scale Humanity's Last Exam Score (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-scale-humanitys-last-exam.png", width=1200, height=800)
