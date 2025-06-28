import pandas as pd
import plotly.graph_objects as go
from pricing import load_pricing_data, get_model_cost

# Create the dataset combining SimpleBench scores with LLM pricing data
# SimpleBench scores from https://simple-bench.com/
# Pricing data loaded from llm-prices.json
#
# COST MULTIPLIERS APPLIED:
# 2x multiplier for thinking models (all except GPT-4.1)

# Load pricing data from JSON file
pricing_data = load_pricing_data()

# Model list with explicit thinking/non-thinking tracking
# Format: (simplebench_name, pricing_base_name, score, is_thinking)
model_data = [
    ("Gemini 2.5 Pro (06-05)", "Gemini 2.5 Pro", 62.4, True),  # Gemini 2.5 IS thinking
    (
        "Claude 4 Opus (thinking)",
        "Claude 4 Opus",
        58.8,
        True,
    ),  # Claude 4 Opus thinking IS thinking
    ("o3 (high)", "o3", 53.1, True),  # o3 IS thinking
    (
        "Claude 4 Sonnet (thinking)",
        "Claude 4 Sonnet",
        45.5,
        True,
    ),  # Claude 4 Sonnet thinking IS thinking
    ("o4-mini (high)", "o4-mini", 38.7, True),  # o4-mini high IS thinking
    ("GPT-4.1", "GPT-4.1", 27.0, False),  # GPT-4.1 is NOT thinking
]

# Extract data for plotting
simplebench_names = []
base_pricing_names = []
scores = []
is_thinking_flags = []

for simplebench_name, pricing_base_name, score, is_thinking in model_data:
    simplebench_names.append(simplebench_name)
    base_pricing_names.append(pricing_base_name)
    scores.append(score)
    is_thinking_flags.append(is_thinking)

# Calculate costs using the pricing module with explicit thinking multiplier
costs = []
for base_name, is_thinking in zip(base_pricing_names, is_thinking_flags):
    base_cost = get_model_cost(base_name, pricing_data)
    if is_thinking:
        # Apply 2x multiplier for thinking models
        costs.append(base_cost * 2)
    else:
        costs.append(base_cost)

data = {
    "model": simplebench_names,
    "score": scores,
    "cost": costs,
}

df = pd.DataFrame(data)

# Identify Pareto frontier
pareto_models = []
df_sorted = df.sort_values("cost")

for idx, row in df_sorted.iterrows():
    dominated = False
    for idx2, row2 in df_sorted.iterrows():
        # Check if another model dominates this one
        # A model is dominated if another model has lower or equal cost AND higher or equal performance
        # For equal cost, we need the higher performance model to dominate
        if (row2["cost"] < row["cost"] and row2["score"] >= row["score"]) or (
            row2["cost"] == row["cost"] and row2["score"] > row["score"]
        ):
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
        y=df["score"],
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
        y=pareto_df["score"],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Pareto Frontier",
    )
)

fig.update_layout(
    title="LLM Pareto Frontier: Cost vs SimpleBench Score",
    xaxis_title="Cost ($) - Average of Input+Output per Million Tokens",
    yaxis_title="SimpleBench Score (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-simplebench.png", width=1200, height=800)
