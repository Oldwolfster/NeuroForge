'''
# LegoManager Plan - Final

## Goal
Single Source of Truth for all legos. No lists to sync. No `var_name` stamping. Legos self-describe, system discovers.

---

## Phase 1: LegoBase Superclass

- Create `_LegoBase.py` in `legos` folder
- `LegoBase` class with:
  - **Required:** `name`, `applicable_dimensions` (enforced non-empty)
        # Option A: Single string (if lego only fits one dimension)
        applicable_dimensions = "hidden_activation"

        # Option B: List (if lego could fit multiple dimensions)
        applicable_dimensions = ["hidden_activation", "output_activation"]

        # Option C: Tuple (immutable, clearer intent)
        applicable_dimensions = ("hidden_activation", "output_activation")

  - **Optional:** `desc`, `when_to_use`, `best_for`
  - `serial_name()` method: `StrategyActivation` → `"Activation_ReLU"`

- Update each Strategy class to subclass `LegoBase`, call `super().__init__()`

---

## Phase 2: LegoManager Discovery

- Create `_LegoManager.py` in `legos` folder
- Scan `legos` folder for all `.py` files without underscore prefix
- Import each module, find all instances of `LegoBase` subclasses
- Build registry dict keyed by dimension name:

registry = {
    "hidden_activation": {
        # ELINIMATE AND ENSURE KEY MATCHES CONFIG - "config_attr": "hidden_activation",
        "lego_kind":   StrategyActivation,
        "members":     [Activation_ReLU, Activation_Sigmoid, ...]
    },
    "optimizer": {
        "config_attr": "optimizer",
        "lego_kind":   StrategyOptimizer,
        "members":     [Optimizer_SGD, Optimizer_Adam, ...]
    },
}

- Build reverse lookup dict during scan: `{"Activation_ReLU": <instance>, ...}`

---

## Phase 3: Serialization

- `lego_to_string(instance)` → calls `instance.serial_name()` → `"Activation_ReLU"`
- `string_to_lego(string)` → reverse lookup dict → `Activation_ReLU` instance

---

## Phase 4: Integration

- `expand_wildcards()` uses `registry[dimension]["members"]`
- `validate_dimensions()` checks `isinstance(value, registry[dimension]["lego_kind"])`
- Delete all the old prefix-matching, `var_name` stamping, synced lists
- `BatchCreator` uses `LegoManager` for everything dimension-related

---

## Non-lego dimensions
`seed`, `architecture`, `learning_rate`, `batch_size` - handle with simple set of known keys, no instantiation needed. Details TBD as we build.

---

## API unchanged
User still writes:

dimensions = {
    "optimizer": [Optimizer_Adam, Optimizer_SGD],
    "hidden_activation": "*",
    "seed": [1, 2, 3],
}
'''


class LegoBase:
    """Base class for all swappable strategy components."""

    def __init__(self, name, applicable_dimensions, desc="", when_to_use="", best_for=""):
        if not applicable_dimensions: raise ValueError(f"LegoBase '{name}' must declare at least one applicable_dimension")
        self.name = name
        self.applicable_dimensions = applicable_dimensions
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for

    def serial_name(self):
        """Returns the variable name assigned during discovery."""
        return self.STAMPED_name
        """Returns serializable string like 'Activation_ReLU'"""
        #prefix = self.__class__.__name__.replace("Strategy", "")
        #return f"{prefix}_{self.name}"

    def __repr__(self):
        return f"Instance[{self.name}]"