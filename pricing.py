import json
from pathlib import Path


def load_pricing_data():
    """Load pricing data from the JSON file and return a dictionary mapping model names to costs."""
    json_path = Path(__file__).parent / "llm-prices.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a mapping from model name to average cost (input + output) / 2
    pricing = {}

    for provider_info in data["models"]:
        for model in provider_info["models"]:
            avg_cost = (model["input_price"] + model["output_price"]) / 2
            pricing[model["name"]] = avg_cost

    return pricing


def get_model_cost(model_name, pricing_data):
    """Get the cost for a model."""

    # Direct lookup in pricing data
    if model_name in pricing_data:
        return pricing_data[model_name]

    # Return 0 for models not found in pricing data
    return 0
