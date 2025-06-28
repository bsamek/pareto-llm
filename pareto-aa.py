import pandas as pd
import plotly.graph_objects as go

# Create the dataset from Artificial Analysis data
# Intelligence scores and costs from https://artificialanalysis.ai/
# Note: Using "Blended USD/1M Tokens" cost and "Intelligence Index" from AA
#
# Only including specified models:
# - o3 (not o3-mini or o3-pro)
# - o4-mini
# - gpt-4o (chatgpt)
# - gpt-4.1
# - All Claude 4 models
# - All Gemini 2.5 models

model_data = [
    # OpenAI models
    ("o3", 70, 3.50),
    ("o4-mini (high)", 70, 1.93),
    ("GPT-4o (ChatGPT)", 40, 7.50),
    ("GPT-4.1", 53, 3.50),
    # Claude 4 models
    ("Claude 4 Opus Thinking", 64, 30.00),
    ("Claude 4 Sonnet Thinking", 61, 6.00),
    ("Claude 4 Opus", 58, 30.00),
    ("Claude 4 Sonnet", 53, 6.00),
    # Gemini 2.5 models
    ("Gemini 2.5 Pro", 70, 3.44),
    ("Gemini 2.5 Pro (Mar '25)", 69, 3.44),
    ("Gemini 2.5 Pro (May '25)", 68, 3.44),
    ("Gemini 2.5 Flash (Reasoning)", 65, 0.99),
    ("Gemini 2.5 Flash (April '25) (Reasoning)", 60, 0.99),
    ("Gemini 2.5 Flash-Lite (Reasoning)", 55, 0.17),
    ("Gemini 2.5 Flash", 53, 0.26),
    ("Gemini 2.5 Flash-Lite", 46, 0.17),
]

# Extract data for plotting
models = []
intelligence_scores = []
costs = []

for model, intelligence, cost in model_data:
    models.append(model)
    intelligence_scores.append(intelligence)
    costs.append(cost)

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
