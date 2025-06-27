import pandas as pd
import plotly.graph_objects as go

# Create the dataset
data = {
    "model": [
        "gemini-2.5-pro-preview-06-05 (32k think)",
        "o3 (high) + gpt-4.1",
        "o3 (high)",
        "gemini-2.5-pro-preview-06-05 (default think)",
        "o3",
        "Gemini 2.5 Pro Preview 05-06",
        "claude-opus-4-20250514 (32k thinking)",
        "o4-mini (high)",
        "DeepSeek R1 (0528)",
        "claude-opus-4-20250514 (no think)",
        "claude-3-7-sonnet-20250219 (32k thinking tokens)",
        "DeepSeek R1 + claude-3-5-sonnet-20241022",
        "o1-2024-12-17 (high)",
        "claude-sonnet-4-20250514 (32k thinking)",
        "claude-3-7-sonnet-20250219 (no thinking)",
        "o3-mini (high)",
        "DeepSeek R1",
        "claude-sonnet-4-20250514 (no thinking)",
        "gemini-2.5-flash-preview-05-20 (24k think)",
        "DeepSeek V3 (0324)",
        "o3-mini (medium)",
        "Grok 3 Beta",
        "gpt-4.1",
        "claude-3-5-sonnet-20241022",
        "Grok 3 Mini Beta (high)",
        "DeepSeek Chat V3 (prev)",
        "gemini-2.5-flash-preview-04-17 (default)",
        "chatgpt-4o-latest (2025-03-29)",
        "gpt-4.5-preview",
        "gemini-2.5-flash-preview-05-20 (no think)",
        "Qwen3 32B",
        "Grok 3 Mini Beta (low)",
        "o1-mini-2024-09-12",
        "gpt-4.1-mini",
        "claude-3-5-haiku-20241022",
        "chatgpt-4o-latest (2025-02-15)",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "DeepSeek Chat V2.5",
        "Codestral 25.01",
        "gpt-4.1-nano",
        "gpt-4o-mini-2024-07-18",
    ],
    "accuracy": [
        83.1,
        82.7,
        81.3,
        79.1,
        76.9,
        76.9,
        72.0,
        72.0,
        71.4,
        70.7,
        64.9,
        64.0,
        61.7,
        61.3,
        60.4,
        60.4,
        56.9,
        56.4,
        55.1,
        55.1,
        53.8,
        53.3,
        52.4,
        51.6,
        49.3,
        48.4,
        47.1,
        45.3,
        44.9,
        44.0,
        40.0,
        34.7,
        32.9,
        32.4,
        28.0,
        27.1,
        23.1,
        18.2,
        17.8,
        11.1,
        8.9,
        3.6,
    ],
    "cost": [
        49.88,
        69.29,
        21.23,
        45.6,
        13.75,
        37.41,
        65.75,
        19.64,
        4.8,
        68.63,
        36.83,
        13.29,
        186.5,
        26.58,
        17.72,
        18.16,
        5.42,
        15.82,
        8.56,
        1.12,
        8.86,
        11.03,
        9.86,
        14.41,
        0.73,
        0.34,
        1.85,
        19.74,
        183.18,
        1.14,
        0.76,
        0.79,
        18.58,
        1.99,
        6.06,
        14.37,
        7.03,
        6.74,
        0.51,
        1.98,
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
        mode="markers",
        marker=dict(
            color="blue",
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
    title="LLM Pareto Frontier: Cost vs Accuracy",
    xaxis_title="Cost ($)",
    yaxis_title="Accuracy (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
