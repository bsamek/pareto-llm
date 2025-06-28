import pandas as pd
import plotly.graph_objects as go

# Create the dataset combining LiveBench Coding scores with LLM pricing data
# LiveBench Coding Average scores from https://livebench.ai/ (2025-05-30)
# Pricing data from https://www.llm-prices.com/
# Using average of input+output costs for simplicity
#
# COST MULTIPLIERS APPLIED:
# 1. Thinking token cost adjustments (simplified to consistent 2x):
#    - All thinking models: 2x base cost vs non-thinking
#    - Applied to: Claude Opus/Sonnet Thinking, Gemini Pro Max Thinking, Gemini Flash
#
# 2. High vs Medium model cost adjustments:
#    - "High" models: 2x cost vs "Medium" models
#    - Applied to: o3 High vs o3 Medium, o4-Mini High vs o4-Mini Medium

data = {
    "model": [
        "o3 High",
        "o3 Medium",
        "o4-Mini High",
        "o4-Mini Medium",
        "Claude 4 Opus Thinking",
        "Claude 4 Opus",
        "Claude 4 Sonnet Thinking",
        "Claude 4 Sonnet",
        "Gemini 2.5 Pro Preview (2025-06-05 Max Thinking)",
        "Gemini 2.5 Pro Preview (2025-06-05)",
        "Gemini 2.5 Flash Preview (2025-05-20)",
        "GPT-4.1",
        "ChatGPT-4o",
        "GPT-4.1 Mini",
        "GPT-4o",
        "GPT-4.1 Nano",
    ],
    "accuracy": [
        76.71,  # o3 High
        77.86,  # o3 Medium
        79.98,  # o4-Mini High
        74.22,  # o4-Mini Medium
        73.25,  # Claude 4 Opus Thinking
        73.58,  # Claude 4 Opus
        73.58,  # Claude 4 Sonnet Thinking
        78.25,  # Claude 4 Sonnet
        73.90,  # Gemini 2.5 Pro Preview Max Thinking
        70.70,  # Gemini 2.5 Pro Preview
        63.53,  # Gemini 2.5 Flash Preview
        73.19,  # GPT-4.1
        77.48,  # ChatGPT-4o
        72.11,  # GPT-4.1 Mini
        69.29,  # GPT-4o
        63.92,  # GPT-4.1 Nano
    ],
    "cost": [
        10.0,  # o3 High: (2+8)/2 * 2 (high multiplier vs medium)
        5.0,  # o3 Medium: (2+8)/2
        5.5,  # o4-Mini High: (1.1+4.4)/2 * 2 (high multiplier vs medium)
        2.75,  # o4-Mini Medium: (1.1+4.4)/2
        90.0,  # Claude 4 Opus Thinking: (15+75)/2 * 2 (thinking multiplier)
        45.0,  # Claude 4 Opus: (15+75)/2
        18.0,  # Claude 4 Sonnet Thinking: (3+15)/2 * 2 (thinking multiplier)
        9.0,  # Claude 4 Sonnet: (3+15)/2
        12.5,  # Gemini 2.5 Pro Preview Max Thinking: (1.25+10)/2 * 2 (thinking multiplier)
        6.25,  # Gemini 2.5 Pro Preview: (1.25+10)/2
        2.8,  # Gemini 2.5 Flash Preview: (0.3+2.5)/2 * 2 (thinking multiplier)
        5.0,  # GPT-4.1: (2+8)/2
        10.0,  # ChatGPT-4o Latest: (5+15)/2
        1.0,  # GPT-4.1 Mini: (0.4+1.6)/2
        6.25,  # GPT-4o: (2.5+10)/2
        0.25,  # GPT-4.1 Nano: (0.1+0.4)/2
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
    title="LLM Pareto Frontier: Cost vs LiveBench Coding Score",
    xaxis_title="Cost ($) - Average of Input+Output per Million Tokens",
    yaxis_title="LiveBench Coding Score (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
fig.write_image("pareto-livebench-coding.png", width=1200, height=800)
