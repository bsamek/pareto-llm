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

    # Direct lookup in pricing data first
    if model_name in pricing_data:
        return pricing_data[model_name]

    # Fallback for models not found (shouldn't happen with updated JSON)
    model_mapping = {
        # These should now be in the JSON, but keeping as fallbacks
        "o3": 5.0,
        "o4-mini": 2.75,
        "GPT 4.1": 5.0,
        "GPT 4.1 Mini": 1.0,
        "GPT 4.1 Nano": 0.25,
        "ChatGPT 4o Latest": 10.0,
        "Claude 4 Opus": 45.0,
        "Claude 4 Sonnet": 9.0,
        "Gemini 2.5 Pro": 6.25,
        "Gemini 2.5 Flash": 1.40,
        "Gemini 2.5 Flash-Lite Preview": 0.25,
    }

    # Fallback to model_mapping if not in pricing_data
    return model_mapping.get(model_name, 0)
