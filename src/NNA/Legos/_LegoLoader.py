# _LegoUtils.py (lives in src/NNA/legos/)

from pathlib import Path
import importlib
import re

LEGOS_DIR = Path(__file__).parent

# Only for dimension keys that don't match file prefix because there are multiple versions
DIMENSION_ALIASES = {
    "hidden_activation": "Activation",
    "output_activation": "Activation",
    "target_scaler": "Scaler",
    "input_scalers": "Scaler",
}


def get_all_legos(dimension_key: str) -> list:
    """Wildcard expansion: 'initializer' -> [Initializer_Xavier, Initializer_He, ...]"""
    prefix = DIMENSION_ALIASES.get(dimension_key, dimension_key.capitalize())
    lego_file = LEGOS_DIR / f"{prefix}.py"

    if not lego_file.exists():
        raise ValueError(f"No lego file: {lego_file}")

    pattern = re.compile(rf'^({prefix}_\w+)\s*=', re.MULTILINE)
    with open(lego_file, 'r') as f:
        matches = pattern.findall(f.read())

    module = importlib.import_module(f".{prefix}", package="src.NNA.legos")
    return [getattr(module, name) for name in matches]


def get_lego_by_name(instance_name: str):
    """DB deserialization: 'Initializer_Xavier' -> Initializer_Xavier instance"""
    prefix = instance_name.split('_')[0]
    module = importlib.import_module(f".{prefix}", package="src.NNA.legos")
    return getattr(module, instance_name)