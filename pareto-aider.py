import pandas as pd
import plotly.graph_objects as go

# Create the dataset
data = {
    "model": [
        "gemini-2.5-pro-preview-06-05 (32k think)",
        "o3 (high)",
        "gemini-2.5-pro-preview-06-05 (default)",
        "o3",
        "Gemini 2.5 Pro Preview 05-06",
        "claude-opus-4-20250514 (32k thinking)",
        "claude-opus-4-20250514 (no think)",
        "claude-sonnet-4-20250514 (32k thinking)",
        "claude-sonnet-4-20250514 (no thinking)",
        "gemini-2.5-flash-preview-05-20 (24k think)",
        "gpt-4.1",
        "gpt-4.5-preview",
        "chatgpt-4o-latest (2025-03-29)",
        "gemini-2.5-flash-preview-05-20 (no think)",
        "gpt-4.1-mini",
        "chatgpt-4o-latest (2025-02-15)",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4.1-nano",
        "gpt-4o-mini-2024-07-18",
    ],
    "accuracy": [
        83.1,
        81.3,
        79.1,
        76.9,
        76.9,
        72.0,
        70.7,
        61.3,
        56.4,
        55.1,
        52.4,
        44.9,
        45.3,
        44.0,
        32.4,
        27.1,
        23.1,
        18.2,
        8.9,
        3.6,
    ],
    "cost": [
        49.88,
        21.23,
        45.60,
        13.75,
        37.41,
        65.75,
        68.63,
        26.58,
        15.82,
        8.56,
        9.86,
        183.18,
        19.74,
        1.14,
        1.99,
        14.37,
        7.03,
        6.74,
        0.43,
        0.32,
    ],
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
    title="LLM Pareto Frontier: Cost vs Accuracy (Aider)",
    xaxis_title="Cost ($)",
    yaxis_title="Accuracy (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-aider.png", width=1200, height=800)
