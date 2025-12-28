# _LegoUtils.py (lives in src/NNA/legos/)

from pathlib import Path
import importlib


class LegoLoader:

    def __init__(self):
        self.legos_dir = Path(__file__).parent
        self.dimension_aliases = {
            "hidden_activation": "Activation",
            "output_activation": "Activation",
            "target_scaler": "Scaler",
            "input_scalers": "Scaler",
        }



    def get_all_legos(self, dimension_key: str) -> list:
        """Wildcard expansion: 'initializer' -> [Initializer_Xavier, Initializer_He, ...]"""
        prefix = self.dimension_aliases.get(dimension_key, dimension_key.capitalize())
        lego_file = self.legos_dir / f"{prefix}.py"

        if not lego_file.exists():          raise ValueError(f"No lego file: {lego_file}")

        instance_names = self.scan_for_instances(lego_file, prefix)
        module = importlib.import_module(f".{prefix}", package="src.NNA.Legos")

        instances = []
        for name in instance_names:
            instance = getattr(module, name)
            instance.var_name = name  # Stamp it
            instances.append(instance)

        return instances

    def scan_for_instances(self, lego_file: Path, prefix: str) -> list:
        """Find all 'Prefix_Something = ' declarations in file"""
        instances = []
        target = f"{prefix}_"

        with open(lego_file, 'r') as f:
            for line in f:
                stripped = line.lstrip()
                if not stripped.startswith(target):
                    continue
                # Find the instance name (everything before ' =' or '=')
                for i, char in enumerate(stripped):
                    if char in ' =':
                        instance_name = stripped[:i]
                        instances.append(instance_name)
                        break

        return instances

    def get_lego_by_name(self, instance_name: str):
        """DB deserialization: 'Initializer_Xavier' -> Initializer_Xavier instance"""
        prefix = instance_name.split('_')[0]
        module = importlib.import_module(f".{prefix}", package="src.NNA.legos")
        return getattr(module, instance_name)

    # _LegoLoader.py

    def stamp_var_name(self, dimension_key: str, instance):
        """Ensure instance has var_name attribute by reverse lookup"""
        if hasattr(instance, 'var_name'):
            return instance

        prefix = self.dimension_aliases.get(dimension_key, dimension_key.capitalize())
        lego_file = self.legos_dir / f"{prefix}.py"

        instance_names = self.scan_for_instances(lego_file, prefix)
        module = importlib.import_module(f".{prefix}", package="src.NNA.Legos")

        for name in instance_names:
            module_instance = getattr(module, name, None)
            if module_instance is None:
                continue
            if module_instance is instance:
                instance.var_name = name
                return instance

        return instance