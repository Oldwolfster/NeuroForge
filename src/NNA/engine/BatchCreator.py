
from src.ArenaSettings import HyperParameters
import random
from src.NNA.legos._LegoManager import LegoManager
from src.NNA.utils.db_prep_disk import check_batch_schema
from pathlib import Path
from itertools import product

class BatchCreator:
    def __init__(self, hyper: HyperParameters):
        self.hyper          = hyper
        self.conn           = hyper.db_dsk.conn
        check_batch_schema  ( hyper.db_dsk.conn)
        self.lego_mgr       = LegoManager()

    def create_new_batch(self):
        """Create a brand new batch - will fail if run_ids collide."""
        dimensions = self.prep_dimensions()
        dimensions = self.expand_wildcards(dimensions)
        dim_keys = list(dimensions.keys())
        dim_values = [dimensions[k] for k in dim_keys]
        batch_id = self.save_batch()
        print(f"Creating NEW batch #{batch_id}")
        run_number = 0

        for gladiator in self.hyper.gladiators:
            lr_flag = self.is_lr_overriding_sweep(gladiator, dim_keys)
            for arena in self.hyper.arenas:
                for combo in product(*dim_values):
                    run_number += 1
                    run_config = dict(zip(dim_keys, combo))
                    run_config['gladiator'] = gladiator
                    run_config['arena'] = arena
                    run_config['lr_specified'] = lr_flag
                    self.save_training_run(run_config, batch_id)
        return batch_id

    def resume_existing_batch(self, batch_id: int):
        """Resume an existing batch - only adds runs that don't exist yet."""
        print(f"Resuming batch #{batch_id}")
        cursor = self.conn.cursor()

        # Get highest run_id so far
        cursor.execute('SELECT COALESCE(MAX(run_id), 0) FROM batch_details WHERE batch_id = ?', (batch_id,))
        next_run_id = cursor.fetchone()[0] + 1

        dimensions = self.prep_dimensions()
        dimensions = self.expand_wildcards(dimensions)
        dim_keys = list(dimensions.keys())
        dim_values = [dimensions[k] for k in dim_keys]

        for gladiator in self.hyper.gladiators:
            lr_flag = self.is_lr_overriding_sweep(gladiator, dim_keys)
            for arena in self.hyper.arenas:
                for combo in product(*dim_values):
                    run_config = dict(zip(dim_keys, combo))
                    run_config['gladiator'] = gladiator
                    run_config['arena'] = arena
                    run_config['lr_specified'] = lr_flag

                    # Check if this config already exists
                    if not self.run_exists(batch_id, run_config):
                        self.save_training_run(run_config, batch_id, next_run_id)
                        next_run_id += 1

        return batch_id

    def run_exists(self, batch_id: int, run_config: dict) -> bool:
        """Check if a run with this exact config already exists."""
        cursor = self.conn.cursor()

        # Build WHERE clause for all config keys
        conditions = []
        params = [batch_id]

        for key, value in run_config.items():
            serialized = self.serialize_value(key, value)
            conditions.append(f"(key = ? AND value = ?)")
            params.extend([key, serialized])

        query = f'''
            SELECT run_id, COUNT(*) as match_count
            FROM batch_details
            WHERE batch_id = ? AND ({' OR '.join(conditions)})
            GROUP BY run_id
            HAVING match_count = ?
        '''
        params.append(len(run_config))
        cursor.execute(query, params)
        return cursor.fetchone() is not None

    def serialize_value(self, key, value):
        """Serialize a value for storage."""
        if key in ['gladiator', 'arena', 'lr_specified']:   return str(value)
        elif self.lego_mgr.is_lego_dimension(key):          return self.lego_mgr.lego_to_string(value)
        else:                                               return str(value)

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

    def save_batch(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO batch_specs (
                batch_name, batch_notes, dimensions, gladiators, arenas
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            self.hyper.batch_name,
            self.hyper.batch_notes,
            str(self.hyper.dimensions),
            str(self.hyper.gladiators),
            str(self.hyper.arenas),
        ))
        self.conn.commit()
        print("Saved batch")
        return cursor.lastrowid

    def save_training_run(self, run_config: dict, batch_id: int):
        """Save run as key-value pairs in batch_details table."""
        cursor = self.conn.cursor()

        # Create history record FIRST - this generates the run_id (PK)
        cursor.execute('''
            INSERT INTO batch_history (batch_id, status, gladiator, arena)
            VALUES (?, 'pending', ?, ?)
        ''', (batch_id, run_config['gladiator'], run_config['arena']))

        run_id = cursor.lastrowid  # Get the auto-generated run_id

        # Now insert config key-value pairs using that run_id
        for key, value in run_config.items():
            serialized = self.serialize_value(key, value)
            cursor.execute('''
                INSERT INTO batch_details (batch_id, run_id, key, value)
                VALUES (?, ?, ?, ?)
            ''', (batch_id, run_id, key, serialized))

        self.conn.commit()

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


