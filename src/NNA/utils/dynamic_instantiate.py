# src/NNA/utils/dynamic_instantiate.py

import importlib.util
import inspect
from pathlib import Path


def dynamic_instantiate(filename, steps_up, path_down, *args):
    """
    Dynamically instantiate a class from a file.

    Args:
        filename: Name of the .py file (without extension)
        steps_up: How many directories up from this file
        path_down: Path down from there to search directory
        *args: Constructor arguments

    Examples:
        dynamic_instantiate("MyArena", 1, "coliseum/arenas", num_samples)
        dynamic_instantiate("AutoForge", 1, "coliseum/gladiators", TRI)
        dynamic_instantiate("SomeReport", 2, "reports", data)
    """
    # Start from this file's location
    base = Path(__file__).parent.resolve()

    # Go up
    for _ in range(steps_up):
        base = base.parent

    # Go down
    search_dir = base / path_down

    if not search_dir.exists():
        raise ImportError(f"Directory not found: {search_dir}")

    # Find the file
    found_files = list(search_dir.rglob(f"{filename}.py"))

    if len(found_files) == 0:
        raise ImportError(f"{filename}.py not found in {search_dir}")
    if len(found_files) > 1:
        raise ValueError(f"Duplicate {filename}.py found: {[str(f) for f in found_files]}")

    found_file = found_files[0]

    # Load directly from file path
    spec = importlib.util.spec_from_file_location(filename, found_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find and instantiate the class defined in this module
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == filename:
            instance = obj(*args)
            try:
                instance.source_code = found_file.read_text(encoding='utf-8')
            except IOError:
                instance.source_code = None
            return instance

    raise ImportError(f"No class found in {filename}.py")
#########################################
# Specific to training data             #
#########################################

# src/NNA/utils/dynamic_instantiate.py (add below dynamic_instantiate)

from src.NNA.engine.TrainingData import TrainingData


def instantiate_arena(arena_name, training_set_size):
    """
    Instantiate an arena and return TrainingData.

    Args:
        arena_name: Name of the arena file (without .py)
        training_set_size: Number of samples to generate
    """
    arena = dynamic_instantiate(arena_name, 1, "coliseum/arenas", training_set_size)
    arena.arena_name = arena_name
    src = arena.source_code

    result = arena.generate_training_data_with_or_without_labels()

    if isinstance(result, tuple):
        data = result[0]
        feature_labels = result[1] if len(result) > 1 else []
        target_labels = result[2] if len(result) > 2 else []
        td = TrainingData(data, feature_labels, target_labels)
    else:
        data = result
        td = TrainingData(data)

    td.source_code = src
    td.arena_name = arena_name
    return td