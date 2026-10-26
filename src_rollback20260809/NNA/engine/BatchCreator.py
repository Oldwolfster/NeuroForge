
from src.ArenaSettings import HyperParameters
import random
from src.NNA.legos._LegoManager import LegoManager

from pathlib import Path
from itertools import product
import json

from src.NNA.utils.enums import RecordLevel


class BatchCreator:
    def __init__(self, hyper: HyperParameters):
        self.hyper          = hyper
        self.conn           = hyper.nf_db.conn  # NEW
        self.lego_mgr       = LegoManager()

    def create_new_batch(self, batch_id=None):
        """batch_id=None: standalone — we mint the batch row. Given: NF minted it, we stamp expected_runs."""
        dimensions = self.prep_dimensions()
        dimensions = self.expand_wildcards(dimensions)
        dim_keys = list(dimensions.keys())
        dim_values = [dimensions[k] for k in dim_keys]

        if batch_id is None:
            batch_id = self.save_batch(dimensions)
        else:
            self.stamp_expected_runs(batch_id, dimensions)
        print(f"Creating NEW batch #{batch_id} in db {self.hyper.nf_db.db_path}")
        for gladiator in self.hyper.gladiators:
            lr_flag = self.is_lr_overriding_sweep(gladiator, dim_keys)
            for arena in self.hyper.arenas:
                for combo in product(*dim_values):
                    run_config = dict(zip(dim_keys, combo))
                    run_config['gladiator'] = gladiator
                    run_config['arena'] = arena
                    run_config['lr_specified'] = lr_flag
                    self.save_training_run(run_config, batch_id)
        return batch_id

    def serialize_value(self, key, value):
        """Column-ready. Naming IS wiring: key IS the batch_history column."""
        if key == 'lr_specified':                    return int(bool(value))
        if isinstance(value, (int, float)):          return value
        if self.lego_mgr.is_lego_dimension(key):     return self.lego_mgr.lego_to_string(value)
        return str(value)                            # gladiator, arena, architecture list

    def prep_dimensions_orig(self):
        dimensions = self.validate_dimensions(self.hyper.dimensions)
        self.ensure_output_neuron(dimensions)
        if "seed" not in dimensions:
            if self.hyper.random_seed != 0: dimensions["seed"] = [self.hyper.random_seed]                 # User specified a specific seed - ignore seed_replicates, use it once
            else:                           dimensions["seed"] = [random.randint(1, 999999) for _ in range(self.hyper.seed_replicates)]                 # User wants randomness (random_seed=0) - honor seed_replicates
        return dimensions

    def prep_dimensions(self):
        dimensions = self.validate_dimensions(self.hyper.dimensions)
        self.ensure_output_neuron(dimensions)

        if "seed" not in dimensions:
            seeds = []
            if self.hyper.random_seed != 0:
                seeds.append(self.hyper.random_seed)
            remaining = self.hyper.seed_replicates - len(seeds)
            seeds.extend([random.randint(1, 999999) for _ in range(remaining)])
            dimensions["seed"] = seeds if seeds else [random.randint(1, 999999)]
        return dimensions

    def ensure_output_neuron(self, dimensions: dict) -> None:
        if "architecture" not in dimensions: return
        for arch in dimensions["architecture"]:
            if arch[-1] != 1: arch.append(1)

    def expand_wildcards(self, dimensions: dict[str, list]) -> dict[str, list]:
        """Expand '*' wildcards to all available legos"""
        expanded = {}
        for key, values in dimensions.items():
            if values == "*":
                # Wildcard - get all legos for this dimension from registry
                if key not in self.lego_mgr.registry: raise ValueError(f"Unknown dimension: {key}")
                expanded[key] = self.lego_mgr.registry[key]["legos"]
            else: expanded[key] = values if isinstance(values, list) else [values] # Pass through as-is (primitives or lego instances)
        return expanded

    def is_lr_overriding_sweep(self,gladiator_name: str, dimension_keys: list) -> bool:
        """
        Check if LR should skip sweep because it's set in gladiator OR dimensions.
        Returns:
            True if LR is explicitly set (skip sweep)
            False if LR should be swept
        """
        if "learning_rate" in dimension_keys:            return True    # Check if LR is in dimensions
        return  self.model_sets_lr(gladiator_name)            # Check if gladiator file sets LR

    def model_sets_lr(self, gladiator_name: str) -> bool:
        """
        Check if gladiator file explicitly sets learning_rate.
        Returns True if uncommented line contains 'config.learning_rate'
        """
        gladiator_dir = Path(__file__).parent.parent / "coliseum" / "gladiators"
        gladiator_file = gladiator_dir / f"{gladiator_name}.py"

        if not gladiator_file.exists():
            print(f"⚠️ Warning: Could not find gladiator file '{gladiator_name}.py'")
            return False

        with open(gladiator_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "config.learning_rate" in line and not line.strip().startswith("#"):
                    return True
        return False

    def save_batch(self, dimensions) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO batch (batch_name, batch_notes, config, expected_runs)
            VALUES (?, ?, ?, ?)
        ''', (
            self.hyper.batch_name,
            self.hyper.batch_notes,
            self.build_config_json(dimensions),
            self.count_expected_runs(dimensions),
        ))
        self.conn.commit()
        print("Saved batch")
        return cursor.lastrowid

    # BatchCreator.py  method: save_training_run  Update: full intent into the pending batch_history row; batch_details is dead
    def save_training_run(self, run_config: dict, batch_id: int):
        """Unknown key = missing column = loud sqlite failure. HOUSTON by design."""
        row             = {key: self.serialize_value(key, value) for key, value in run_config.items()}
        row["batch_id"] = batch_id
        row["status"]   = "pending"
        row["record_level"] = self.record_level()
        cols            = ", ".join(row)
        marks           = ", ".join("?" * len(row))
        self.conn.cursor().execute(f"INSERT INTO batch_history ({cols}) VALUES ({marks})", list(row.values()))
        self.conn.commit()

    def record_level(self):
        """Setup ships 'Full'; the row carries 2. None means no request — display_need decides."""
        name = getattr(self.hyper, "record_level", None)
        return RecordLevel[name.upper()].value if name else None
    def get_valid_config_keys(self) -> set:
        """Get valid dimension keys from Config class attributes."""
        from src.NNA.engine.Config import Config
        excluded = {'TRI', 'scaler'}
        return {attr for attr in dir(Config) if not attr.startswith('_') and attr not in excluded}


    def validate_dimensions(self, dimensions: dict) -> dict:
        """Validate keys and normalize single values to lists"""
        validated = {}

        for key, values in dimensions.items():
            if not self.lego_mgr.is_valid_dimension(key):   raise ValueError(f"Unknown dimension: '{key}'. Valid dimensions: {self.lego_mgr.get_valid_dimensions()}")

            if values == "*":
                if not self.lego_mgr.is_lego_dimension(key):   raise ValueError(f"Cannot use wildcard on primitive dimension '{key}'")
                validated[key] = values
                continue

            # Normalize to list
            values = values if isinstance(values, list) else [values]

            # Validate lego types i.e. not putting a loss function in a weight initializer
            if self.lego_mgr.is_lego_dimension(key):
                expected_kind = self.lego_mgr.registry[key]["kind"]
                for v in values:
                    if not isinstance(v, expected_kind):    raise ValueError(f"Dimension '{key}' expects {expected_kind.__name__}, got {type(v).__name__}")
            validated[key] = values
        return validated


    # BatchCreator.py  method: build_config_json  NEW: same JSON shape NF ships — one frozen truth for both launch paths
    def build_config_json(self, dimensions) -> str:
        dims  = self.serialize_dimensions(dimensions)
        seeds = dims.pop("seed", [])
        return json.dumps({
            "gladiators":        self.hyper.gladiators,
            "arenas":            self.hyper.arenas,
            "seeds":             seeds,
            "epochs_to_run":     self.hyper.epochs_to_run,
            "training_set_size": self.hyper.training_set_size,
            "dimensions":        dims,
        })

    # BatchCreator.py  method: serialize_dimensions  NEW: legos -> names, everything else JSON-native
    def serialize_dimensions(self, dimensions) -> dict:
        return {key: [self.serialize_value(key, v) for v in values]
                for key, values in dimensions.items()}

    # BatchCreator.py  method: count_expected_runs  NEW: gladiators x arenas x every dimension (seed included)
    def count_expected_runs(self, dimensions) -> int:
        total = len(self.hyper.gladiators) * len(self.hyper.arenas)
        for values in dimensions.values(): total *= len(values)
        return total

    # BatchCreator.py  method: stamp_expected_runs  NEW: NF minted the row; the expander owns the truth on run count
    def stamp_expected_runs(self, batch_id, dimensions):
        self.conn.execute("UPDATE batch SET expected_runs = ? WHERE batch_id = ?",
                          (self.count_expected_runs(dimensions), batch_id))
        self.conn.commit()