import pandas as pd
import plotly.graph_objects as go
from pricing import load_pricing_data, get_model_cost

# Create the dataset combining LM Arena Elo scores with LLM pricing data
# LM Arena scores from https://lmarena.ai/ (latest data)
# Pricing data loaded from llm-prices.json
#
# COST MULTIPLIERS APPLIED:
# 1. Thinking token cost adjustments (simplified to consistent 2x):
#    - All thinking models: 2x base cost vs non-thinking
#    - Applied to CoT models

# Load pricing data from JSON file
pricing_data = load_pricing_data()

# Model list and Elo scores - filtered to include only specified models
# Mapping LM Arena model names to our pricing model names
model_data = [
    # OpenAI models
    ("o3-2025-04-16", "o3", 1451),
    ("chatgpt-4o-latest-20250326", "ChatGPT-4o", 1442),
    ("gpt-4.1-2025-04-14", "GPT-4.1", 1411),
    ("o4-mini-2025-04-16", "o4-mini", 1398),
    ("gpt-4.1-mini-2025-04-14", "GPT-4.1 Mini", 1374),
    ("gpt-4.1-nano-2025-04-14", "GPT-4.1 Nano", 1320),
    # Anthropic models
    ("claude-opus-4-20250514", "Claude 4 Opus", 1418),
    ("claude-sonnet-4-20250514", "Claude 4 Sonnet", 1393),
    (
        "claude-3-7-sonnet-20250219-thinking-32k",
        "Claude 4 Sonnet Thinking",
        1387,
    ),  # CoT model
    # Google models
    ("gemini-2.5-pro", "Gemini 2.5 Pro", 1467),
    ("gemini-2.5-flash", "Gemini 2.5 Flash", 1418),
    (
        "gemini-2.5-flash-lite-preview-06-17-thinking",
        "Gemini 2.5 Flash Thinking",
        1387,
    ),  # CoT model
]

# Extract data for plotting
arena_names = []
pricing_names = []
elo_scores = []

for arena_name, pricing_name, elo in model_data:
    arena_names.append(arena_name)
    pricing_names.append(pricing_name)
    elo_scores.append(elo)

# Calculate costs using the pricing module
costs = []
for pricing_name in pricing_names:
    if "Thinking" in pricing_name:
        # Apply 2x multiplier for CoT models
        base_name = pricing_name.replace(" Thinking", "")
        base_cost = get_model_cost(base_name, pricing_data)
        costs.append(base_cost * 2)
    else:
        costs.append(get_model_cost(pricing_name, pricing_data))

data = {
    "model": arena_names,
    "elo_score": elo_scores,
    "cost": costs,
}

df = pd.DataFrame(data)

# Identify Pareto frontier
pareto_models = []
df_sorted = df.sort_values("cost")

for idx, row in df_sorted.iterrows():
    dominated = False
    for idx2, row2 in df_sorted.iterrows():
        if row2["cost"] < row["cost"] and row2["elo_score"] >= row["elo_score"]:
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
        y=df["elo_score"],
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
        textfont=dict(size=8),
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
        y=pareto_df["elo_score"],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Pareto Frontier",
    )
)

fig.update_layout(
    title="LLM Pareto Frontier: Cost vs LM Arena Elo Score",
    xaxis_title="Cost ($) - Average of Input+Output per Million Tokens",
    yaxis_title="LM Arena Elo Score",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-lmarena.png", width=1200, height=800)
