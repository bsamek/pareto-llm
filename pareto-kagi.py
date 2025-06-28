import pandas as pd
import plotly.graph_objects as go

# Create the dataset
data = {
    "model": [
        "o3 [CoT]",
        "claude-4-opus [CoT]",
        "gemini-2-5-pro [CoT]",
        "claude-4-sonnet [CoT]",
        "o4-mini [CoT]",
        "claude-4-opus [no-think]",
        "chatgpt-4o",
        "gpt-4-1",
        "claude-4-sonnet [no-think]",
        "gpt-4-1-mini",
        "gpt-4o",
        "gemini-2-5-flash [no-think]",
    ],
    "accuracy": [
        78.03,
        77.46,
        74.03,
        68.69,
        67.79,
        60.21,
        54.80,
        53.76,
        52.58,
        48.80,
        42.60,
        41.88,
    ],
    "cost": [
        2.08,
        2.84,
        0.19,
        0.62,
        0.32,
        1.06,
        0.57,
        0.18,
        0.23,
        0.05,
        0.22,
        0.02,
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
