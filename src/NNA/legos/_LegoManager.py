import importlib
import os

from src.NNA.legos._LegoBase import LegoBase
from src.NNA.legos.Activation import * #TODO CAN REMOVED - for testing only.


class LegoManager:
    NON_LEGO_DIMENSIONS         = ["seed", "architecture", "learning_rate", "batch_size"]
    SERIALIZATION_EXCLUSIONS    = {'TRI', 'scaler', 'lego_mgr'}

    def __init__(self):
        self.lego_files         = []
        self.lego_instances     = []
        self.registry           = {}
        self.reverse_lookup     = {}

        self.discover_files()
        for lego_socket in self.lego_files: self.load_lego_instances(lego_socket)
        self.build_registry()
        self.build_reverse_lookup()
        self.add_non_lego_dimension()
        self.test()

    def discover_files(self):
        folder              = os.path.dirname(__file__)
        all_files           = os.listdir(folder)
        self.lego_files     = [f for f in all_files if f.endswith('.py') and not f.startswith('_')]
        self.lego_files     = [f[:-3] for f in self.lego_files]

    def load_lego_instances(self, module_name):
        #print(f"First line of load lego {module_name}")
        module = importlib.import_module(f".{module_name}", package=__package__)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, LegoBase) and attr.__class__.__module__ == module.__name__:
                attr.STAMPED_name = attr_name  #print(f"  Found: In {module_name} found  {attr_name} -> {attr}")
                self.lego_instances.append(attr)

    def build_registry(self):
        for lego in self.lego_instances:
            lego_kind = lego.__class__
            for dimension in lego.applicable_dimensions:
                if dimension not in self.registry:
                    self.registry[dimension] = {"kind": lego_kind, "legos": []}
                self.registry[dimension]["legos"].append(lego)

    def build_reverse_lookup(self):
        for lego in self.lego_instances:
            serial = lego.serial_name()
            if serial in self.reverse_lookup:
                raise ValueError(f"Duplicate serial_name: {serial}")
            self.reverse_lookup[serial] = lego

    def add_non_lego_dimension(self):
        # Add non- legos to dict such as seed etc.
        for not_a_lego  in LegoManager.NON_LEGO_DIMENSIONS:
            self.registry[not_a_lego] = {"kind": None, "legos": None}

    def lego_to_string(self, lego_instance):
        return lego_instance.serial_name()

    def string_to_lego(self, serial_name):
        if serial_name not in self.reverse_lookup: raise ValueError(f"Unknown lego: {serial_name}")
        return self.reverse_lookup[serial_name]

    def is_valid_dimension(self, key):
        """Check if dimension key exists in registry"""
        return key in self.registry

    def is_lego_dimension(self, key):
        """Check if dimension supports lego instances (vs primitives)"""
        return key in self.registry and self.registry[key]["legos"] is not None

    def get_valid_dimensions(self):
        """Return sorted list of all valid dimension names"""
        return sorted(self.registry.keys())

    def serialize_ANYTHING(self, obj) -> dict:
        """Serialize any object - converts lego instances to strings."""
        result = {}
        for attr in dir(obj):
            value = getattr(obj, attr, None)
            if self.should_skip(attr, value):                               continue

            if self.is_lego_dimension(attr) and value is not None:
                result[attr] = self.lego_to_string(value)
            elif isinstance(value, list):
                result[attr] = str(value)
            elif hasattr(value, 'name'):  # ← ADD THIS for Scaler fallback
                result[attr] = value.name
            else:
                result[attr] = value
        return result

    # _LegoManager.py - replace serialize_ANYTHING temporarily

    def serialize_ANYTHING(self, obj) -> dict:
        """Serialize any object - converts lego instances to strings."""
        result = {}
        for attr in dir(obj):
            value = getattr(obj, attr, None)

            # DIAGNOSTIC - check our suspect attributes
            if attr in ['weight_initializer', 'hidden_activation', 'output_activation',
                        'loss_function', 'optimizer']:
                skip = self.should_skip(attr, value)
                is_lego = self.is_lego_dimension(attr)
                print(f"  TRACE {attr}: skip={skip}, is_lego={is_lego}, value={value}")

            if self.should_skip(attr, value):                               continue

            if self.is_lego_dimension(attr) and value is not None:
                result[attr] = self.lego_to_string(value)
            elif isinstance(value, list):
                result[attr] = str(value)
            elif hasattr(value, 'name'):
                result[attr] = value.name
            else:
                result[attr] = value
        return result

    def should_skip(self, attr, value):
        """Check ALL exclusion rules - underscores, exclusions list, AND callables."""
        if attr.startswith('_'):                                            return True
        if attr in LegoManager.SERIALIZATION_EXCLUSIONS:                    return True
        if callable(value):                                                 return True
        return False

    # _LegoManager.py - replace should_skip temporarily

    # _LegoManager.py - final should_skip

    def should_skip(self, attr, value):
        """Check ALL exclusion rules - underscores, exclusions list, AND callables."""
        if attr.startswith('_'):                                            return True
        if attr in LegoManager.SERIALIZATION_EXCLUSIONS:                    return True
        if self.is_lego_dimension(attr):                                    return False  # Legos pass even if callable
        if callable(value):                                                 return True
        return False


    def test(self):
        for lego_socket     in self.lego_files:     print(f"lego_socket    {lego_socket}")
        for lego_instances  in self.lego_instances: print(f"lego_instances {lego_instances}")
        print("******************************************************")
        print(self.reverse_lookup.get("Activation_NoDamnFunction"))
        print(self.reverse_lookup.get("Activation_ReLU"))

        print("******************************************************")
        print("REGISTRY BY DIMENSION:")
        for dim, data in self.registry.items():
            if data["legos"] is None:  # Skip primitives
                print(f"  {dim}: [primitive dimension]")
            else:
                print(f"  {dim}:")
                print(f"    kind: {data['kind'].__name__}")
                print(f"    legos: {[lego.serial_name() for lego in data['legos']]}")
        print("******************************************************")

        print("******************************************************")
        print("REVERSE LOOKUP CONTENTS:")
        for key, value in self.reverse_lookup.items():
            print(f"  '{key}' -> {value}")
        print("******************************************************")
        print("SERIALIZATION/DESERIALIZATION TEST:")
        s = self.lego_to_string(Activation_ReLU)
        print(f"Serialized: {s}")
        lego = self.string_to_lego(s)
        print(f"Deserialized: {lego}")


# _LegoManager.py - add this method to LegoManager class

    def diagnose_serialization(self, obj, label=""):
        """Diagnostic: Compare object attributes against registry."""
        print(f"\n{'=' * 60}")
        print(f"SERIALIZATION DIAGNOSTIC: {label}")
        print(f"{'=' * 60}")

        # 1. What dimensions are in the registry?
        print(f"\n[1] REGISTRY DIMENSIONS:")
        for dim in sorted(self.registry.keys()):
            is_lego = self.registry[dim]["legos"] is not None
            print(f"    {dim}: {'LEGO' if is_lego else 'primitive'}")

        # 2. What attributes does the object have?
        print(f"\n[2] OBJECT ATTRIBUTES (type={type(obj).__name__}):")
        suspect_attrs = ['weight_initializer', 'hidden_activation', 'output_activation',
                         'optimizer', 'loss_function', 'initializer']

        for attr in suspect_attrs:
            if hasattr(obj, attr):
                value = getattr(obj, attr, None)
                in_registry = attr in self.registry
                is_lego_dim = self.is_lego_dimension(attr)
                print(f"    {attr}:")
                print(f"        value     = {value}")
                print(f"        type      = {type(value).__name__ if value else 'None'}")
                print(f"        in_registry   = {in_registry}")
                print(f"        is_lego_dim   = {is_lego_dim}")
                if value and hasattr(value, 'serial_name'):
                    print(f"        serial_name() = {value.serial_name()}")
            else:
                print(f"    {attr}: NOT ON OBJECT")

        # 3. What does serialize_ANYTHING produce?
        print(f"\n[3] SERIALIZATION RESULT:")
        result = self.serialize_ANYTHING(obj)
        for key in suspect_attrs:
            if key in result:
                print(f"    {key} = {result[key]}")
            else:
                print(f"    {key} = MISSING FROM RESULT")

        print(f"{'=' * 60}\n")
        return result