import pandas as pd
import plotly.graph_objects as go

# Dataset from Kagi; filtered to only include rows where provider is
# exactly "kagi" or "kagi (ult)" (excludes deprecated, openrouter, Mistral, Nebius, etc.).
data = {
    "model": [
        "claude-4-opus-thinking",
        "grok-4",
        "claude-4-sonnet-thinking",
        "gpt-5",
        "o3-pro",
        "gemini-2-5-pro",
        "gpt-5-mini",
        "deepseek-r1",
        "qwen-3-235b-a22b-thinking",
        "o3",
        "o4-mini",
        "gpt-5-nano",
        "grok-3",
        "grok-3-mini",
        "claude-4-opus",
        "gpt-oss-120b",
        "gemini-2-5-flash-thinking",
        "llama-4-maverick",
        "claude-4-sonnet",
        "qwen-3-235b-a22b (no thinking)",
        "gpt-oss-20b",
        "deepseek chat v3.1",
        "glm-4-5",
        "qwen-3-coder",
        "mistral-medium",
        "kimi-k2",
        "gemini-2-5-flash",
        "gemini-2-5-flash-lite",
        "mistral-small",
    ],
    "%accuracy": [
        74.3,
        73.6,
        73.0,
        72.7,
        72.1,
        70.3,
        70.3,
        69.4,
        69.4,
        67.6,
        67.6,
        62.2,
        61.3,
        61.3,
        59.6,
        58.6,
        56.8,
        55.9,
        55.9,
        55.0,
        53.2,
        53.2,
        52.3,
        49.5,
        45.9,
        45.0,
        44.1,
        40.5,
        37.8,
    ],
    "Cost($)": [
        22.4,
        1.0,
        5.4,
        7.1,
        34.2,
        1.7,
        4.9,
        9.9,
        0.1,
        4.8,
        3.1,
        0.4,
        2.6,
        0.3,
        8.4,
        0.4,
        0.5,
        0.2,
        1.8,
        0.4,
        0.5,
        0.4,
        5.2,
        0.8,
        0.3,
        1.1,
        0.4,
        0.1,
        0.1,
    ],
}

df = pd.DataFrame(data)
df.rename(columns={"%accuracy": "accuracy", "Cost($)": "cost"}, inplace=True)

# Filter out models with no cost data or zero/negative cost
df = df[df["cost"].notna()]
df = df[df["cost"] > 0].copy()

# For each cost, find the model with the highest accuracy
df_best_at_cost = df.loc[df.groupby("cost")["accuracy"].idxmax()]

# Identify Pareto frontier from these best models
df_sorted = df_best_at_cost.sort_values(by=["cost", "accuracy"], ascending=[True, True])
pareto_models = []
last_best_accuracy = -1

for _, row in df_sorted.iterrows():
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
fig.write_image("pareto-kagi.png", width=1200, height=800)
