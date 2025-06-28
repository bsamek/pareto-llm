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
    """Get the cost for a model, applying multipliers for CoT and High variants."""

    # Map model names to their base pricing data equivalents
    model_mapping = {
        # OpenAI models (hypothetical future models not in JSON yet)
        "o3": 5.0,  # Placeholder - not in current JSON
        "o4-mini": 2.75,  # Placeholder - not in current JSON
        "GPT-4.1": 5.0,  # Placeholder - not in current JSON
        "GPT-4.1 Mini": 1.0,  # Placeholder - not in current JSON
        "GPT-4.1 Nano": 0.25,  # Placeholder - not in current JSON
        "ChatGPT-4o": 10.0,  # Placeholder - not in current JSON
        # Models available in JSON
        "Claude 4 Opus": pricing_data.get("Claude 4 Opus", 45.0),
        "Claude 4 Sonnet": pricing_data.get("Claude 4 Sonnet", 9.0),
        "Gemini 2.5 Pro": pricing_data.get("Gemini 2.5 Pro", 6.25),
        "Gemini 2.5 Flash": pricing_data.get("Gemini 2.5 Flash", 1.40),
        "GPT-4o": pricing_data.get("GPT-4o", 6.25),
    }

    # Extract base model name and determine multipliers
    base_cost = 0
    multiplier = 1

    # Handle specific model patterns
    if "o3 High" in model_name:
        base_cost = model_mapping["o3"]
        multiplier = 2  # High multiplier
    elif "o3 Medium" in model_name or "o3" in model_name:
        base_cost = model_mapping["o3"]
    elif "o4-Mini High" in model_name:
        base_cost = model_mapping["o4-mini"]
        multiplier = 2  # High multiplier
    elif "o4-Mini Medium" in model_name or "o4-Mini" in model_name:
        base_cost = model_mapping["o4-mini"]
    elif "Claude 4 Opus Thinking" in model_name:
        base_cost = model_mapping["Claude 4 Opus"]
        multiplier = 2  # CoT multiplier
    elif "Claude 4 Opus" in model_name:
        base_cost = model_mapping["Claude 4 Opus"]
    elif "Claude 4 Sonnet Thinking" in model_name:
        base_cost = model_mapping["Claude 4 Sonnet"]
        multiplier = 2  # CoT multiplier
    elif "Claude 4 Sonnet" in model_name:
        base_cost = model_mapping["Claude 4 Sonnet"]
    elif "Gemini 2.5 Pro Preview" in model_name and "Thinking" in model_name:
        base_cost = model_mapping["Gemini 2.5 Pro"]
        multiplier = 2  # CoT multiplier
    elif "Gemini 2.5 Pro Preview" in model_name:
        base_cost = model_mapping["Gemini 2.5 Pro"]
    elif "Gemini 2.5 Flash Preview" in model_name:
        base_cost = model_mapping["Gemini 2.5 Flash"]
        multiplier = 2  # CoT multiplier (Flash is thinking model)
    elif (
        "GPT-4.1" in model_name
        and "Mini" not in model_name
        and "Nano" not in model_name
    ):
        base_cost = model_mapping["GPT-4.1"]
    elif "GPT-4.1 Mini" in model_name:
        base_cost = model_mapping["GPT-4.1 Mini"]
    elif "GPT-4.1 Nano" in model_name:
        base_cost = model_mapping["GPT-4.1 Nano"]
    elif "ChatGPT-4o" in model_name:
        base_cost = model_mapping["ChatGPT-4o"]
    elif "GPT-4o" in model_name:
        base_cost = model_mapping["GPT-4o"]

    return base_cost * multiplier
