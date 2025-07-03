import pandas as pd
import plotly.graph_objects as go
from pricing import load_pricing_data, get_model_cost

# Create the dataset combining Scale MultiChallenge scores with LLM pricing data
# Scale MultiChallenge scores from https://scale.com/leaderboard/multichallenge
# Pricing data loaded from llm-prices.json
#
# COST MULTIPLIERS APPLIED:
# 1. Thinking token cost adjustments (simplified to consistent 2x):
#    - All thinking models: 2x base cost vs non-thinking
#    - Applied to: Claude Opus/Sonnet Thinking, Gemini Pro/Flash, o3, o4-mini
#
# 2. High vs Medium model cost adjustments:
#    - "High" models: 2x cost vs "Medium" models
#    - Applied to: o3 High vs o3 Medium, o4-Mini High vs o4-Mini Medium

# Load pricing data from JSON file
pricing_data = load_pricing_data()

# Model list and accuracy scores
models = [
    "o3",
    "o3",
    "o4-mini",
    "o4-mini",
    "Gemini 2.5 Pro",
    "Gemini 2.5 Flash Preview",
    "Claude 4 Opus",
    "Claude 4 Sonnet",
    "GPT 4.1",
]

accuracy_scores = [
    56.51,  # o3 High
    59.09,  # o3 Medium
    42.99,  # o4-Mini High
    43.83,  # o4-Mini Medium
    49.91,  # Gemini 2.5 Pro
    52.62,  # Gemini 2.5 Flash Preview
    53.90,  # Claude 4 Opus
    53.12,  # Claude 4 Sonnet
    38.26,  # GPT 4.1
]

# Calculate costs using the pricing module
# Apply cost multipliers for thinking models and high/medium variants
costs = []
display_names = []

for i, model in enumerate(models):
    base_cost = get_model_cost(model, pricing_data)

    # Apply cost multipliers based on the original model names
    if i == 0:  # o3 High (thinking model with high variant)
        cost = base_cost * 2 * 2  # 2x for thinking, 2x for high
        display_name = "o3 High"
    elif i == 1:  # o3 Medium (thinking model)
        cost = base_cost * 2  # 2x for thinking
        display_name = "o3 Medium"
    elif i == 2:  # o4-Mini High (thinking model with high variant)
        cost = base_cost * 2 * 2  # 2x for thinking, 2x for high
        display_name = "o4-Mini High"
    elif i == 3:  # o4-Mini Medium (thinking model)
        cost = base_cost * 2  # 2x for thinking
        display_name = "o4-Mini Medium"
    elif i == 4:  # Gemini 2.5 Pro (thinking model)
        cost = base_cost * 2  # 2x for thinking
        display_name = "Gemini 2.5 Pro"
    elif i == 5:  # Gemini Flash (thinking)
        cost = base_cost * 2
        display_name = "Gemini 2.5 Flash Preview"
    elif i == 6:  # Claude Opus Thinking
        cost = base_cost * 2
        display_name = "Claude 4 Opus Thinking"
    elif i == 7:  # Claude Sonnet Thinking
        cost = base_cost * 2
        display_name = "Claude 4 Sonnet Thinking"
    else:
        cost = base_cost
        display_name = model

    costs.append(cost)
    display_names.append(display_name)

data = {
    "model": display_names,
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
    title="LLM Pareto Frontier: Cost vs Scale MultiChallenge Score",
    xaxis_title="Cost ($) - Average of Input+Output per Million Tokens",
    yaxis_title="Scale MultiChallenge Score (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-scale-multichallenge.png", width=1200, height=800)
