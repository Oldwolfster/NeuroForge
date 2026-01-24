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

        instances = []
        seen_names = set()

        for lego_file in sorted(self.legos_dir.glob("*.py")):
            if lego_file.name.startswith("_"):
                continue
            if lego_file.name == "__init__.py":
                continue

            instance_names = self.scan_for_instances(lego_file, prefix)
            if not instance_names:
                continue

            module = importlib.import_module(f".{lego_file.stem}", package="src.NNA.legos")

            for name in instance_names:
                if name in seen_names:
                    continue
                seen_names.add(name)

                instance = getattr(module, name)
                instance.var_name = name  # Stamp it
                instances.append(instance)

        if not instances:
            raise ValueError(f"No legos found for prefix '{prefix}_' in {self.legos_dir}")

        return instances

    def scan_for_instances(self, lego_file: Path, prefix: str) -> list:
        """Find all 'Prefix_Something = ' declarations in file"""
        instances = []
        target = f"{prefix}_"
        try:
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

        except UnicodeDecodeError:
            # Fallback to latin-1 which accepts all bytes
            with open(lego_file, 'r', encoding='latin-1') as f:
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
        module = importlib.import_module(f".{prefix}", package="src.NNA.legos")

        for name in instance_names:
            module_instance = getattr(module, name, None)
            if module_instance is None:
                continue
            if module_instance is instance:
                instance.var_name = name
                return instance

        return instance