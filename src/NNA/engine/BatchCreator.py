from src.ArenaSettings import HyperParameters
import random
from src.NNA.legos._LegoWildCard import LegoLoader
from pathlib import Path
from itertools import product

class BatchCreator:
    def __init__(self, hyper: HyperParameters):
        self.hyper = hyper
        self.conn = hyper.db_dsk.conn
        self.check_schema()

    def create_a_batch(self):
        dimensions  = self.prep_dimensions()
        dimensions  = self.expand_wildcards(dimensions)
        dim_keys    = list(dimensions.keys())
        dim_values  = [dimensions[k] for k in dim_keys]
        batch_id    = self.save_batch()
        print(f"Creating batch #{batch_id}")
        run_number  = 0

        for gladiator in self.hyper.gladiators:
            lr_flag = self.is_lr_overriding_sweep(gladiator, dim_keys)
            for arena in self.hyper.arenas:
                for combo in product(*dim_values):
                    run_number += 1
                    config = dict(zip(dim_keys, combo))
                    config['gladiator'] = gladiator
                    config['arena'] = arena
                    config['lr_specified'] = lr_flag
                    #print(config)
                    #print(f"{run_number}: {gladiator} | {arena} | {config.get('initializer', '-')}")
                    self.save_training_run(config, batch_id)
        return batch_id

    def prep_dimensions(self):
        dimensions = self.validate_dimensions(self.hyper.dimensions)
        self.ensure_output_neuron(dimensions)
        if "seed" not in dimensions:
            if self.hyper.random_seed != 0: dimensions["seed"] = [self.hyper.random_seed]                 # User specified a specific seed - ignore seed_replicates, use it once
            else:                           dimensions["seed"] = [random.randint(1, 999999) for _ in range(self.hyper.seed_replicates)]                 # User wants randomness (random_seed=0) - honor seed_replicates
        return dimensions

    def ensure_output_neuron(self, dimensions: dict) -> None:
        if "architecture" not in dimensions: return
        for arch in dimensions["architecture"]:
            if arch[-1] != 1: arch.append(1)

    # BatchCreator.py

    def expand_wildcards(self, dimensions: dict[str, list]) -> dict[str, list]:
        """Expand '*' wildcards to all available legos"""
        loader = LegoLoader()
        expanded = {}
        for key, values in dimensions.items():
            if values == "*":
                expanded[key] = loader.get_all_legos(key)
            elif isinstance(values, list):
                expanded[key] = values
            else:
                expanded[key] = [values]
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

    def check_schema(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name TEXT,
                batch_notes TEXT,
                dimensions TEXT,
                full_record_count INTEGER DEFAULT 2,
                parent_batch_id INTEGER,
                gladiator TEXT,
                arena TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER,
                status TEXT DEFAULT 'pending',
                record_level TEXT,
                seed INTEGER,
                gladiator TEXT,
                arena TEXT,
                architecture TEXT,
                loss TEXT,
                optimizer TEXT,
                hidden_activation TEXT,
                output_activation TEXT,
                initializer TEXT,
                input_scalers TEXT,
                output_scaler TEXT,
                learning_rate REAL,
                lr_specified INTEGER,
                batch_size INTEGER,
                roi_mode TEXT,
                epoch_count INTEGER,
                regularization TEXT,
                dropout REAL,
                momentum REAL,
                accuracy REAL,
                final_mae REAL,
                best_mae REAL,
                convergence_condition TEXT,
                problem_type TEXT,
                sample_count INTEGER,
                target_min REAL,
                target_max REAL,
                target_min_label TEXT,
                target_max_label TEXT,
                target_mean REAL,
                target_stdev REAL,
                runtime_seconds REAL,
                timestamp DATETIME,
                notes TEXT
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_batch_status_run
            ON training_runs(batch_id, status)
        ''')
        self.conn.commit()

    def save_batch(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO batches (
                batch_name, batch_notes, dimensions, full_record_count, gladiator, arena
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.hyper.batch_name,
            self.hyper.batch_notes,
            str(self.hyper.dimensions),
            0, #we don't know this at time of generating it.
            str(self.hyper.gladiators),
            str(self.hyper.arenas),
        ))
        self.conn.commit()
        return cursor.lastrowid

    def save_training_run(self, config: dict, batch_id: int):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO training_runs (
                batch_id, status, seed,
                gladiator, arena, architecture,
                loss, optimizer, hidden_activation, output_activation,
                initializer, learning_rate, lr_specified
            ) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            batch_id,
            config.get('seed'),
            config.get('gladiator'),
            config.get('arena'),
            str(config.get('architecture')),
            getattr(config.get('loss'), 'var_name', None),
            getattr(config.get('optimizer'), 'var_name', None),
            getattr(config.get('hidden_activation'), 'var_name', None),
            getattr(config.get('output_activation'), 'var_name', None),
            getattr(config.get('initializer'), 'var_name', None),
            config.get('learning_rate'),
            config.get('lr_specified'),
        ))
        self.conn.commit()
        return cursor.lastrowid


    VALID_DIMENSION_KEYS = {
        # Lego types
        "loss", "optimizer", "initializer",
        "hidden_activation", "output_activation",
        "input_scalers", "output_scaler",
        # Non-lego dimensions
        "seed", "architecture", "batch_size", "learning_rate",
    }

    def validate_dimensions(self, dimensions: dict) -> dict:
        """Validate keys, normalize single values to lists, check lego types"""
        loader = LegoLoader()
        validated = {}

        lego_keys = {"loss", "optimizer", "initializer",
                     "hidden_activation", "output_activation",
                     "input_scalers", "output_scaler"}

        for key, values in dimensions.items():
            if key not in self.VALID_DIMENSION_KEYS:
                raise ValueError(f"Invalid dimension key: '{key}'. Valid keys: {sorted(self.VALID_DIMENSION_KEYS)}")

            if values == "*":
                validated[key] = values
                continue

            if not isinstance(values, list):
                values = [values]

            if key in lego_keys:
                expected_prefix = self.get_expected_prefix(key)
                stamped = []
                for v in values:
                    loader.stamp_var_name(key, v)
                    if not getattr(v, 'var_name', '').startswith(expected_prefix):
                        raise ValueError(f"Dimension '{key}' expects {expected_prefix}* legos, got: {v.var_name}")
                    stamped.append(v)
                values = stamped

            validated[key] = values

        return validated
    def get_expected_prefix(self, dimension_key: str) -> str:
        """Return expected instance prefix for lego dimensions, None for non-lego"""
        aliases = {
            "hidden_activation": "Activation_",
            "output_activation": "Activation_",
            "input_scalers": "Scaler_",
            "output_scaler": "Scaler_",
        }
        if dimension_key in aliases:
            return aliases[dimension_key]
        if dimension_key in {"loss", "optimizer", "initializer"}:
            return dimension_key.capitalize() + "_"
        return None  # Non-lego dimension