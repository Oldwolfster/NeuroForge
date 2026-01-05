# NeuroForge Optimizer Architecture
## Complete Technical Documentation - "Every Detail Exposed"

---

## 1. ARCHITECTURAL OVERVIEW

### 1.1 The Dual System Architecture

NeuroForge operates as two interconnected systems:
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK ARENA (NNA)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │  Gladiator  │───>│   Config    │───>│  Optimizer (Strategy)       │ │
│  │  (Model)    │    │  (Settings) │    │  - compute_adjustment_fn    │ │
│  └─────────────┘    └─────────────┘    │  - get_state_fn             │ │
│         │                              │  - state_headers/operators  │ │
│         v                              └─────────────────────────────┘ │
│  ┌─────────────┐    ┌─────────────┐                 │                  │
│  │   Neuron    │───>│ VCRRecorder │─────────────────┘                  │
│  │  (State)    │    │  (Capture)  │                                    │
│  └─────────────┘    └─────────────┘                                    │
│         │                  │                                           │
└─────────│──────────────────│───────────────────────────────────────────┘
          │                  │
          │                  v
          │         ┌───────────────┐
          │         │   SQLite DB   │
          │         │  - Neurons    │
          │         │  - Weights    │
          │         │  - WtAdj_upd  │
          │         │  - WtAdj_fin  │
          │         │  - ErrorSig   │
          └────────>│  - Samples    │
                    └───────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────────────────┐
│                         NEUROFORGE (NF)                                │
│  ┌─────────────┐    ┌─────────────────────┐    ┌───────────────────┐   │
│  │     VCR     │───>│ DisplayModel__Neuron│───>│   Popup/Tooltip   │   │
│  │  (Playback) │    │     _Base           │    │  (Audit Display)  │   │
│  └─────────────┘    └─────────────────────┘    └───────────────────┘   │
│                              │                                         │
│                              v                                         │
│                     ┌───────────────────┐                              │
│                     │  Optimizer        │                              │
│                     │  _backprop_popup_ │                              │
│                     │  headers_*        │                              │
│                     └───────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Optimizer's Unique Burden

Unlike other "Legos" (Activations, Initializers, Loss Functions, Scalers) which are **self-contained**, 
optimizers have **tentacles** reaching into multiple system components:

| Component          | What Optimizer Touches                                      |
|--------------------|-------------------------------------------------------------|
| **Neuron**         | State arrays: `m[]`, `v[]`, `t`, `accumulated_accepted_blame[]`, `learning_rates[]` |
| **VCRRecorder**    | `record_weight_updates()` with variable-length rows         |
| **DB Schema**      | `WeightAdjustments_update_*`, `WeightAdjustments_finalize_*` with `arg_1..arg_N` |
| **Popup Rendering**| Header arrays determine column count and labels             |

---

## 2. THE STRATEGY PATTERN IMPLEMENTATION

### 2.1 StrategyOptimizer Class Structure
```python
class StrategyOptimizer:
    def __init__(self,
                 name: str,                    # Display name ("Adam", "SGD")
                 desc: str,                    # Full description
                 compute_adjustment_fn,        # Core math: (neuron, weight_id, avg_blame, lr) → adjustment
                 get_state_fn,                 # Audit trail: (neuron, weight_id) → [values...]
                 state_headers: list,          # Column headers for optimizer-specific data
                 state_operators: list,        # Operators between columns (for display)
                 when_to_use: str = "",
                 best_for: str = ""):
```

### 2.2 The Two Functions Every Optimizer Must Provide

#### `compute_adjustment_fn(neuron, weight_id, avg_blame, lr) → float`
The actual weight update math. Called once per weight during batch finalization.

#### `get_state_fn(neuron, weight_id) → list`
Returns the audit trail values. **Must match `state_headers` in count and order.**

### 2.3 Example: SGD (Simplest Case)
```python
def sgd_compute_adjustment(neuron, weight_id, avg_blame, lr):
    return lr * avg_blame

def sgd_get_state(neuron, weight_id):
    return [neuron.learning_rates[weight_id]]

Optimizer_SGD = StrategyOptimizer(
    name="Stochastic Gradient Descent",
    compute_adjustment_fn=sgd_compute_adjustment,
    get_state_fn=sgd_get_state,
    state_headers=["Lrn Rt"],
    state_operators=[" "]
)
```

### 2.4 Example: Adam (Complex Case)
```python
def adam_compute_adjustment(neuron, weight_id, avg_blame, lr):
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    
    neuron.t += 1
    neuron.m[weight_id] = beta1 * neuron.m[weight_id] + (1 - beta1) * avg_blame
    neuron.v[weight_id] = beta2 * neuron.v[weight_id] + (1 - beta2) * (avg_blame ** 2)
    
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
    v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)
    
    return lr * m_hat / (v_hat ** 0.5 + epsilon)

def adam_get_state(neuron, weight_id):
    # Must return 6 values to match state_headers
    return [
        neuron.m[weight_id],      # m
        neuron.v[weight_id],      # v
        neuron.t,                 # t
        m_hat,                    # m_hat (computed)
        v_hat,                    # v_hat (computed)
        scaled_lr                 # Scaled LR (computed)
    ]

Optimizer_Adam = StrategyOptimizer(
    ...
    state_headers=["m", "v", "t", "m_hat", "v_hat", "Scaled LR"],
    state_operators=[" ", " ", " ", " ", " ", " "]
)
```

---

## 3. THE COMPLETE DATA FLOW

### 3.1 Training Flow (NNA Side)
```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SAMPLE PROCESSING                                                           │
│                                                                              │
│  for each sample:                                                            │
│    1. forward_pass() → prediction                                            │
│    2. judge_pass() → loss, loss_gradient                                     │
│    3. back_pass():                                                           │
│       a. determine_blame_for_output_neuron(loss_gradient)                    │
│       b. determine_blame_for_hidden_neurons()                                │
│       c. spread_the_blame() → calls optimizer.update()                       │
│                                      │                                       │
│                                      v                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  optimizer.update(neuron, input_vector, blame, t, config, ...)       │   │
│  │                                                                       │   │
│  │  Calls: universal_batch_update(neuron, input_vector, blame, t, cfg)  │   │
│  │         ┌────────────────────────────────────────────────────────┐   │   │
│  │         │ for each weight_id, input_x:                           │   │   │
│  │         │   raw_adjustment = input_x * accepted_blame            │   │   │
│  │         │   neuron.accumulated_accepted_blame[weight_id] += raw  │   │   │
│  │         │   log: [weight_id, input_x, blame, raw_adj, (cum)?]    │   │   │
│  │         └────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │  Returns: [[epoch, sample, nid, wt_id, batch_id, arg1, arg2, ...]]   │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          v                                                   │
│  VCR.record_weight_updates(logs, "update") → WeightAdjustments_update_X     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Batch Finalization Flow
```
┌──────────────────────────────────────────────────────────────────────────────┐
│  BATCH FINALIZATION (triggered by VCR.maybe_finalize_batch)                 │
│                                                                              │
│  Condition: sample % batch_size == 0  OR  sample == total_samples           │
│                                                                              │
│  optimizer.finalizer(batch_size, epoch, sample, batch_id)                    │
│                           │                                                  │
│                           v                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  universal_batch_finalize(batch_size, optimizer)                      │   │
│  │                                                                       │   │
│  │  for each layer, for each neuron, for each weight_id:                │   │
│  │    ┌────────────────────────────────────────────────────────────┐    │   │
│  │    │ blame_total = neuron.accumulated_accepted_blame[weight_id] │    │   │
│  │    │ avg_blame = blame_total / batch_size                       │    │   │
│  │    │ lr = neuron.learning_rates[weight_id]                      │    │   │
│  │    │                                                            │    │   │
│  │    │ adjustment = optimizer.compute_adjustment(                 │    │   │
│  │    │                  neuron, weight_id, avg_blame, lr)         │    │   │
│  │    │                                                            │    │   │
│  │    │ state = optimizer.get_state(neuron, weight_id)             │    │   │
│  │    │                                                            │    │   │
│  │    │ neuron.weights[weight_id] -= adjustment  ← WEIGHT UPDATED  │    │   │
│  │    │                                                            │    │   │
│  │    │ log: [nid, wt_id, (batch_tot, count, avg)?, ...state]     │    │   │
│  │    └────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  │  Reset: neuron.accumulated_accepted_blame = [0.0] * len(...)         │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          v                                                   │
│  VCR.record_weight_updates(logs, "finalize") → WeightAdjustments_finalize_X │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. DATABASE SCHEMA

### 4.1 Tables Created Per Run
```sql
-- Created by create_weight_adjustments_table(db, run_id, "update", arg_count=12)
CREATE TABLE WeightAdjustments_update_{run_id} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch        INTEGER NOT NULL,
    sample       INTEGER NOT NULL,
    nid          INTEGER NOT NULL,
    weight_index INTEGER NOT NULL,
    batch_id     INTEGER NOT NULL DEFAULT 0,
    arg_1        REAL DEFAULT NULL,  -- Varies by optimizer
    arg_2        REAL DEFAULT NULL,
    arg_3        REAL DEFAULT NULL,
    ...
    arg_12       REAL DEFAULT NULL
);

-- Same structure for finalize
CREATE TABLE WeightAdjustments_finalize_{run_id} ( ... );
```

### 4.2 What Goes in Each Table

#### `WeightAdjustments_update_{run_id}`
Per-sample blame accumulation. Written during `spread_the_blame()`.

| Batch Mode | Columns (arg_1, arg_2, ...) |
|------------|----------------------------|
| Single     | `[input_x, blame, raw_adjustment]` |
| Batch      | `[input_x, blame, raw_adjustment, cumulative]` |

#### `WeightAdjustments_finalize_{run_id}`
Per-batch weight updates. Written at batch boundaries.

| Batch Mode | Columns (arg_1, arg_2, ...) |
|------------|----------------------------|
| Single     | `[...optimizer_state]` |
| Batch      | `[batch_total, count, avg_blame, ...optimizer_state]` |

---

## 5. HEADER SYSTEM (THE DISPLAY CONTRACT)

### 5.1 The Four Header Arrays

Each optimizer instance maintains these after `configure_optimizer()`:
```python
# For update table (blame accumulation)
_backprop_popup_headers_single   = ["Input", "Blame", " Leverage"]
_backprop_popup_operators_single = ["*", "=", " "]

_backprop_popup_headers_batch    = ["Input", "Blame", " Leverage", "Cum."]
_backprop_popup_operators_batch  = ["*", "=", " ", "||"]

# For finalize table (weight adjustment)
_backprop_popup_headers_finalizer   # Built dynamically
_backprop_popup_operators_finalizer # Built dynamically
```

### 5.2 Header Assembly in `configure_optimizer()`
```python
def configure_optimizer(self, config):
    if config.batch_size == 1:
        # Single-sample: Just optimizer state
        self._backprop_popup_headers_finalizer = self._state_headers
        self._backprop_popup_operators_finalizer = self._state_operators
    else:
        # Batch mode: Batch mechanics + optimizer state
        self._backprop_popup_headers_finalizer = (
            universal_batch_headers +    # ["BatchTot", "Count", "Avg"]
            self._state_headers          # ["m", "v", "t", ...] (optimizer-specific)
        )
        self._backprop_popup_operators_finalizer = (
            universal_batch_operators +  # ["/", "=", "*"]
            self._state_operators
        )
```

### 5.3 How NeuroForge Reads Headers

In `DisplayModel__Neuron_Base.tooltip_columns_for_backprop_update()`:
```python
if is_batch:
    headers = self.config.optimizer._backprop_popup_headers_batch
    operators = self.config.optimizer._backprop_popup_operators_batch
else:
    headers = self.config.optimizer._backprop_popup_headers_single
    operators = self.config.optimizer._backprop_popup_operators_single

num_args = len(headers)
arg_fields = [f"arg_{i+1}" for i in range(num_args)]  # Dynamically build SELECT
```

---

## 6. NEURON STATE REQUIREMENTS

### 6.1 State Arrays Each Optimizer May Need

| Optimizer    | Needs `m[]` | Needs `v[]` | Needs `t` | Notes |
|--------------|-------------|-------------|-----------|-------|
| SGD          | ❌          | ❌          | ❌        | Stateless |
| Simplex      | ❌          | ❌          | ❌        | Uses `learning_rates[]` |
| Momentum     | ✅          | ❌          | ❌        | Velocity |
| RMSprop      | ❌          | ✅          | ❌        | Squared gradients |
| Adam         | ✅          | ✅          | ✅        | Full state |
| AdaGrad      | ❌          | ✅          | ❌        | Cumulative |
| NAdam        | ✅          | ✅          | ✅        | Like Adam |
| AdamW        | ✅          | ✅          | ✅        | Like Adam |
| Adadelta     | ✅          | ✅          | ❌        | Repurposes m/v |
| AdaMax       | ✅          | ✅          | ✅        | Like Adam |
| MagDetox     | ❌          | ❌          | ❌        | Stateless |

### 6.2 Universal State (All Optimizers)
```python
# In Neuron.__init__():
self.learning_rates = [learning_rate] * len(self.weights)
self.accumulated_accepted_blame = [0.0] * len(self.weights)
```

### 6.3 ⚠️ CURRENT GAP: Optimizer State Initialization

The `m`, `v`, `t` arrays are **commented out** in Neuron.py but **used** by optimizers.
This must be resolved - either:
- Uncomment in Neuron (wasteful for SGD)
- Lazy-initialize in optimizer on first access
- Optimizer declares required state, Neuron initializes conditionally

---

## 7. COMPLETE OPTIMIZER INVENTORY

| Optimizer      | State Headers                              | State Size |
|----------------|-------------------------------------------|------------|
| SGD            | `["Lrn Rt"]`                              | 1 |
| Simplex        | `["Lrn Rt"]`                              | 1 |
| Momentum       | `["velocity", "Scaled LR"]`               | 2 |
| RMSprop        | `["v", "sqrt(v)", "Scaled LR"]`           | 3 |
| Adam           | `["m", "v", "t", "m_hat", "v_hat", "Scaled LR"]` | 6 |
| AdaGrad        | `["G", "sqrt(G)", "Scaled LR"]`           | 3 |
| NAdam          | `["m", "v", "t", "m_hat", "v_hat", "Scaled LR"]` | 6 |
| AdamW          | `["m", "v", "m_hat", "v_hat", "Scaled LR", "WD"]` | 6 |
| Adadelta       | `["Grad²", "Δ²", "RMS(g)", "RMS(Δ)", "Adaptive LR"]` | 5 |
| AdaMax         | `["m", "u_∞", "t", "m_hat", "Scaled LR"]` | 5 |
| MagDetox       | `["Fixed Base α"]`                        | 1 |

---

## 8. THE POPUP RENDERING PIPELINE

### 8.1 Column Assembly Order
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  POPUP COLUMNS (left to right)                                              │
│                                                                             │
│  FORWARD PASS          │  BACKWARD PASS                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  [Cog] [Input] [Weight] [=] [Product]  │  UPDATE COLS  │  FINALIZE  │ STD  │
│                                        │               │   COLS     │ COLS │
│                                        │ Input*Blame   │ (if batch) │ Adj  │
│                                        │ =Leverage     │ m,v,t...   │Before│
│                                        │ (Cum)?        │            │After │
│                                        │               │            │      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  + BLAME SOURCES SECTION (below divider)                                    │
│  - Output: "This is where blame originates..."                              │
│  - Hidden: ErrorSignalCalcs table data                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Data Flow for Popup Generation
```python
# In DisplayModel__Neuron_Base.tooltip_generate_text():

self.tooltip_columns.clear()
self.tooltip_columns.extend(self.tooltip_columns_for_forward_pass())   # 6 cols
self.tooltip_columns.extend(self.tooltip_columns_for_backprop())       # Variable

# tooltip_columns_for_backprop() returns:
#   update_cols          (3-4 cols based on batch mode)
#   + finalize_cols      (optimizer state_headers count)
#   + std_finalize_cols  (3 cols: Adj, Before, After)
#   + error_sig_cols     (injected via helper)
```

---

## 9. KNOWN ISSUES / TECH DEBT

### 9.1 Neuron State Initialization Gap
- `m`, `v`, `t` are commented out in Neuron.py
- Optimizers that need them will crash
- Need lazy init or optimizer-declared requirements

### 9.2 `configure_optimizer` Timing
```python
# NOTE: THIS RUNS AFTER TRAINING!!! BAH!!!!! FIXME
def configure_optimizer(self, config):
```
The comment indicates this runs too late - should run before training to set up headers.

### 9.3 Table-Per-Weight Structure
- Current: `WeightAdjustments_finalize_{run_id}` with one row per weight
- Clean slate goal: Single table with all weights
- Schema is identical per weight, just different `weight_index` values

### 9.4 Hardcoded arg_count
```python
create_weight_adjustments_table(db, run_id, "update", arg_count=12)
```
Fixed at 12 args - what if optimizer needs more?

---

## 10. SUMMARY: THE CONTRACT

For an optimizer to be fully integrated:

1. **Define the math**: `compute_adjustment_fn(neuron, weight_id, avg_blame, lr) → float`

2. **Define the audit trail**: `get_state_fn(neuron, weight_id) → list` 
   - Length MUST equal `len(state_headers)`

3. **Declare display columns**: `state_headers` and `state_operators`

4. **Declare neuron state needs**: (currently implicit, should be explicit)
   - Which of `m[]`, `v[]`, `t` does this optimizer need?

5. **The system handles**:
   - Blame accumulation (`accumulated_accepted_blame`)
   - Batch averaging
   - DB logging with dynamic column count
   - Popup rendering with dynamic headers