import pandas as pd
import plotly.graph_objects as go
from pricing import load_pricing_data, get_model_cost

# Create the dataset from Artificial Analysis data
# Intelligence scores from https://artificialanalysis.ai/
# Costs from llm-prices.json with explicit thinking/non-thinking tracking
#
# Only including specified models:
# - o3 (not o3-mini or o3-pro)
# - o4-mini
# - gpt-4o (chatgpt)
# - gpt-4.1
# - All Claude 4 models
# - All Gemini 2.5 models

# Load pricing data from JSON file
pricing_data = load_pricing_data()

# Model list with explicit thinking/non-thinking tracking
# Format: (display_name, pricing_key, intelligence_score, is_thinking)
model_data = [
    # OpenAI models
    ("o3", "o3", 70, True),  # o3 IS thinking
    ("o4-mini (high)", "o4-mini", 70, True),  # o4-mini IS thinking
    ("GPT-4o (ChatGPT)", "ChatGPT 4o Latest", 40, False),  # ChatGPT is NOT thinking
    ("GPT-4.1", "GPT 4.1", 53, False),  # GPT models are NOT thinking
    # Claude 4 models - NONE are thinking based on AA data
    (
        "Claude 4 Opus Thinking",
        "Claude 4 Opus",
        64,
        True,
    ),  # Claude 4 Opus Thinking IS thinking
    (
        "Claude 4 Sonnet Thinking",
        "Claude 4 Sonnet",
        61,
        True,
    ),  # Claude 4 Sonnet Thinking IS thinking
    ("Claude 4 Opus", "Claude 4 Opus", 58, False),  # Claude 4 Opus is NOT thinking
    (
        "Claude 4 Sonnet",
        "Claude 4 Sonnet",
        53,
        False,
    ),  # Claude 4 Sonnet is NOT thinking
    # Gemini 2.5 models - reasoning variants ARE thinking
    (
        "Gemini 2.5 Pro",
        "Gemini 2.5 Pro",
        70,
        False,
    ),  # Base Gemini 2.5 Pro is NOT thinking
    (
        "Gemini 2.5 Flash (Reasoning)",
        "Gemini 2.5 Flash",
        65,
        True,
    ),  # Reasoning IS thinking
    (
        "Gemini 2.5 Flash (April '25) (Reasoning)",
        "Gemini 2.5 Flash",
        60,
        True,
    ),  # Reasoning IS thinking
    (
        "Gemini 2.5 Flash-Lite (Reasoning)",
        "Gemini 2.5 Flash-Lite Preview",
        55,
        True,
    ),  # Reasoning IS thinking
    ("Gemini 2.5 Flash", "Gemini 2.5 Flash", 53, False),  # Base Flash is NOT thinking
    (
        "Gemini 2.5 Flash-Lite",
        "Gemini 2.5 Flash-Lite Preview",
        46,
        False,
    ),  # Base Flash-Lite is NOT thinking
]

# Extract data for plotting
models = []
intelligence_scores = []
costs = []
is_thinking_flags = []

for display_name, pricing_key, intelligence, is_thinking in model_data:
    models.append(display_name)
    intelligence_scores.append(intelligence)
    is_thinking_flags.append(is_thinking)

    # Calculate cost using the pricing module with explicit thinking multiplier
    base_cost = get_model_cost(pricing_key, pricing_data)
    if is_thinking:
        # Apply 2x multiplier for thinking models
        costs.append(base_cost * 2)
    else:
        costs.append(base_cost)

data = {
    "model": models,
    "intelligence_score": intelligence_scores,
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
        if (
            row2["cost"] < row["cost"]
            and row2["intelligence_score"] >= row["intelligence_score"]
        ) or (
            row2["cost"] == row["cost"]
            and row2["intelligence_score"] > row["intelligence_score"]
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
        y=df["intelligence_score"],
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
        y=pareto_df["intelligence_score"],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Pareto Frontier",
    )
)

fig.update_layout(
    title="LLM Pareto Frontier: Cost vs Artificial Analysis Intelligence Index",
    xaxis_title="Cost ($) - Blended per Million Tokens",
    yaxis_title="Artificial Analysis Intelligence Index",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-aa.png", width=1200, height=800)
