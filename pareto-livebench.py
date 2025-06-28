import pandas as pd
import plotly.graph_objects as go

# Create the dataset combining LiveBench scores with LLM pricing data
# LiveBench Global Average scores from https://livebench.ai/ (2025-05-30)
# Pricing data from https://www.llm-prices.com/
# Using average of input+output costs for simplicity

data = {
    "model": [
        "o3 Pro High",
        "o3 High",
        "Claude 4 Opus Thinking",
        "Gemini 2.5 Pro Preview (2025-05-06)",
        "Claude 4 Sonnet Thinking",
        "o3 Medium",
        "o4-Mini High",
        "Gemini 2.5 Pro Preview (2025-06-05 Max Thinking)",
        "DeepSeek R1 (2025-05-28)",
        "Gemini 2.5 Pro Preview (2025-06-05)",
        "Claude 3.7 Sonnet Thinking",
        "o4-Mini Medium",
        "Claude 4 Opus",
        "DeepSeek R1",
        "Qwen 3 235B A22B",
        "Gemini 2.5 Flash Preview (2025-05-20)",
        "Qwen 3 32B",
        "Claude 4 Sonnet",
        "Gemini 2.5 Flash Preview (2025-04-17)",
        "Grok 3 Mini Beta (High)",
        "GPT-4.5 Preview",
        "Claude 3.7 Sonnet",
        "Grok 3 Beta",
        "GPT-4.1",
        "ChatGPT-4o",
        "Claude 3.5 Sonnet",
        "GPT-4.1 Mini",
        "GPT-4o",
        "Mistral Large",
        "Claude 3.5 Haiku",
        "GPT-4.1 Nano",
    ],
    "accuracy": [
        74.72,
        74.61,
        72.93,
        72.09,
        72.08,
        71.98,
        71.52,
        70.95,
        70.10,
        69.39,
        67.43,
        66.87,
        65.93,
        65.15,
        64.93,
        64.42,
        63.71,
        63.37,
        62.80,
        62.36,
        58.65,
        58.48,
        56.05,
        55.90,
        54.74,
        51.80,
        51.57,
        47.43,
        43.31,
        39.51,
        40.51,
    ],
    "cost": [
        50.0,  # o3 Pro High: (20+80)/2
        5.0,  # o3 High: (2+8)/2
        45.0,  # Claude 4 Opus Thinking: (15+75)/2
        6.25,  # Gemini 2.5 Pro Preview: (1.25+10)/2 (assuming ≤200k)
        9.0,  # Claude 4 Sonnet Thinking: (3+15)/2
        5.0,  # o3 Medium: (2+8)/2
        2.75,  # o4-Mini High: (1.1+4.4)/2
        6.25,  # Gemini 2.5 Pro Preview Max Thinking: (1.25+10)/2
        1.37,  # DeepSeek R1: (0.55+2.19)/2
        6.25,  # Gemini 2.5 Pro Preview: (1.25+10)/2
        9.0,  # Claude 3.7 Sonnet Thinking: (3+15)/2
        2.75,  # o4-Mini Medium: (1.1+4.4)/2
        45.0,  # Claude 4 Opus: (15+75)/2
        1.37,  # DeepSeek R1: (0.55+2.19)/2
        6.25,  # Qwen 3 (assuming similar to Gemini Pro pricing)
        1.55,  # Gemini 2.5 Flash Preview: (0.3+2.5)/2
        6.25,  # Qwen 3 32B (assuming similar to Gemini Pro pricing)
        9.0,  # Claude 4 Sonnet: (3+15)/2
        1.55,  # Gemini 2.5 Flash Preview: (0.3+2.5)/2
        0.4,  # Grok 3 Mini Beta: (0.3+0.5)/2
        112.5,  # GPT-4.5 Preview: (75+150)/2
        9.0,  # Claude 3.7 Sonnet: (3+15)/2
        9.0,  # Grok 3 Beta: (3+15)/2
        5.0,  # GPT-4.1: (2+8)/2
        10.0,  # ChatGPT-4o Latest: (5+15)/2
        9.0,  # Claude 3.5 Sonnet: (3+15)/2
        1.0,  # GPT-4.1 Mini: (0.4+1.6)/2
        6.25,  # GPT-4o: (2.5+10)/2
        4.0,  # Mistral Large: (2+6)/2
        2.4,  # Claude 3.5 Haiku: (0.8+4)/2
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
    title="LLM Pareto Frontier: Cost vs LiveBench Global Score",
    xaxis_title="Cost ($) - Average of Input+Output per Million Tokens",
    yaxis_title="LiveBench Global Score (%)",
    xaxis_type="log",
    showlegend=True,
    legend=dict(
        itemsizing="constant",
    ),
)

fig.show()
