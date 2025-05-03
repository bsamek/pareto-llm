import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create the dataset
data = {
    'model': ['o3', 'claude-3-7-extended-thinking', 'gemini-2-5-pro', 'qwen-qwq-32b', 'o1', 'o3-mini', 'deepseek-r1', 'o4-mini', 'grok-3-mini', 'deepseek-r1-distill-llama-70b', 'gpt-4-1', 'chatgpt-4o', 'deepseek', 'grok-3', 'llama-4-maverick', 'o1-pro', 'gpt-4-1-mini', 'claude-3-7-sonnet', 'claude-3-opus', 'claude-3-sonnet-v2', 'mistral-large', 'claude-3-sonnet-v1', 'mistral-small', 'llama-3-405b', 'llama-3-70b', 'gemini-flash', 'llama-4-scout', 'gpt-4-turbo', 'gpt-4o-mini', 'claude-3-haiku', 'nova-pro', 'gemini-pro-1-5', 'gpt-4-1-nano', 'nova-lite', 'llama-3-3b', 'mistral-nemo'],
    'CoT': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'N', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],
    'accuracy': [76.29, 71.34, 68.72, 65.94, 65.44, 65.16, 64.06, 62.27, 59.17, 54.41, 54.17, 53.09, 50.34, 50.34, 46.09, 44.38, 44.06, 42.94, 41.57, 41.22, 39.42, 37.12, 36.47, 35.85, 34.78, 34.4, 33.43, 32.51, 30.98, 29.63, 29.12, 28.99, 27.83, 26.09, 17.58, 14.37],
    'cost': [2.57191, 2.20567, 0.257, 0.11994, 6.55213, 0.52675, 1.16229, 0.41746, 0.07626, 0.40643, 0.2275, 0.72127, 0.32012, 0.92201, 0.04311, 59.5752, 0.05571, 0.30431, 1.94389, 0.26061, 0.12415, 0.33792, 0.00382, 0.33991, 0.10628, 0.0128, 0.02634, 1.37861, 0.02758, 0.12328, 0.15737, 0.50243, 0.01078, 0.01007, 0.01212, 0.00128]
}

df = pd.DataFrame(data)

# Identify Pareto frontier
pareto_models = []
df_sorted = df.sort_values('cost')

for idx, row in df_sorted.iterrows():
    dominated = False
    for idx2, row2 in df_sorted.iterrows():
        if row2['cost'] < row['cost'] and row2['accuracy'] >= row['accuracy']:
            dominated = True
            break
    if not dominated:
        pareto_models.append(row['model'])

# Create the plot
plt.figure(figsize=(12, 8))

# Plot all models with CoT-based coloring
for idx, row in df.iterrows():
    color = 'red' if row['CoT'] == 'Y' else 'blue'
    marker = 's' if row['model'] in pareto_models else 'o'  # Square for frontier, circle for others
    size = 120 if row['model'] in pareto_models else 60
    edgecolor = 'black' if row['model'] in pareto_models else 'none'
    linewidth = 2 if row['model'] in pareto_models else 0
    
    plt.scatter(row['cost'], row['accuracy'], color=color, marker=marker, s=size, 
                zorder=5 if row['model'] in pareto_models else 3,
                edgecolor=edgecolor, linewidth=linewidth, alpha=0.8)
    
    # Add labels with appropriate formatting based on frontier status
    if row['model'] in pareto_models:
        plt.text(row['cost'], row['accuracy'] + 0.5, row['model'], fontsize=8, 
                ha='center', rotation=45, fontweight='bold')
    else:
        plt.text(row['cost'], row['accuracy'] + 0.5, row['model'], fontsize=6, 
                ha='center', alpha=0.7, rotation=45)

# Connect Pareto frontier points
pareto_df = df[df['model'].isin(pareto_models)].sort_values('cost')
plt.plot(pareto_df['cost'], pareto_df['accuracy'], 'k-', linewidth=2, alpha=0.5, zorder=4)

plt.xscale('log')
plt.xlabel('Cost ($)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('LLM Pareto Frontier: Cost vs Accuracy', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='CoT Models'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Non-CoT Models'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, label='Pareto Frontier', linewidth=1),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other Models')
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()