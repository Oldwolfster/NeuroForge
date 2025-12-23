from enum import IntEnum, auto

from src.NNA.engine import Neuron

# ==============================================================================
# UNIVERSAL COMPONENTS (same for all gradient-based optimizers)
# ==============================================================================

# Blame calculation columns (always shown)
universal_blame_headers_single = ["Input", "Blame", " Leverage"]
universal_blame_operators_single = ["*", "=", " "]

universal_blame_headers_batch = ["Input", "Blame", " Leverage", "Cum."]
universal_blame_operators_batch = ["*", "=", " ", "||"]

# Batch mechanics columns (shown when batching is used)
universal_batch_headers = ["BatchTot", "Count", "Avg"]
universal_batch_operators = ["/", "=", "*"]

# Universal adjustment columns (always shown)
universal_adjust_headers = ["Adj", "Before", "After"]


#class BatchMode(IntEnum):
#    SINGLE_SAMPLE = auto()  # One sample at a time, fixed order
#    MINI_BATCH = auto()  # Mini-batches in fixed order
#    FULL_BATCH = auto()  # All samples per update (no shuffling)


# ==============================================================================
# UNIVERSAL BATCH HANDLING (works for any optimizer)
# ==============================================================================

def universal_batch_update(neuron, input_vector, accepted_blame, t, config):
    """
    Universal update function for all gradient-based optimizers.
    Handles accumulation of blame across batch samples.
    """
    logs = []

    for weight_id, input_x in enumerate(input_vector):
        raw_adjustment = input_x * accepted_blame  # "Wt Blame"
        neuron.accumulated_accepted_blame[weight_id] += raw_adjustment

        # Log appropriate columns based on batch size
        if config.batch_size == 1:
            # Single-sample: 3 columns
            logs.append([weight_id, input_x, accepted_blame, raw_adjustment])
        else:
            # Batch mode: 4 columns (includes cumulative)
            logs.append([weight_id, input_x, accepted_blame, raw_adjustment,
                         neuron.accumulated_accepted_blame[weight_id]])

    return logs


def universal_batch_finalize(batch_size, optimizer):
    """
    Universal finalize function for all gradient-based optimizers.
    Handles averaging and calls optimizer-specific adjustment logic.
    """
    logs = []
    is_batch = batch_size > 1

    for layer in Neuron.layers:
        for neuron in layer:
            for weight_id in range(len(neuron.accumulated_accepted_blame)):
                # Universal: Average the accumulated blame
                blame_total = neuron.accumulated_accepted_blame[weight_id]
                avg_blame = blame_total / batch_size
                lr = neuron.learning_rates[weight_id]

                # Optimizer-specific: Compute adjustment
                adjustment = optimizer.compute_adjustment(neuron, weight_id, avg_blame, lr)

                # Optimizer-specific: Get state for logging
                state = optimizer.get_state(neuron, weight_id)

                # Apply the adjustment
                if weight_id == 0:
                    neuron.bias -= adjustment
                else:
                    neuron.weights[weight_id - 1] -= adjustment

                # Log based on mode
                if is_batch:
                    # Batch mode: [neuron_id, weight_id, batch_total, count, avg] + optimizer_state
                    logs.append([neuron.nid, weight_id, blame_total, batch_size, avg_blame] + state)
                else:
                    # Single-sample: [neuron_id, weight_id] + optimizer_state (skip redundant batch mechanics)
                    logs.append([neuron.nid, weight_id] + state)

            # Reset accumulator for next batch
            neuron.accumulated_accepted_blame = [0.0] * len(neuron.accumulated_accepted_blame)

    return logs

# ==============================================================================
# STRATEGY OPTIMIZER CLASS
# ==============================================================================

class StrategyOptimizer:
    """
    Represents an optimization algorithm.

    Optimizers only need to define:
    1. How to compute adjustment given averaged blame
    2. What state to log for debugging
    3. Display headers for their state
    """

    def __init__(self,
                 name: str,
                 desc: str,
                 compute_adjustment_fn,
                 get_state_fn,
                 state_headers: list,
                 state_operators: list,
                 when_to_use: str = "",
                 best_for: str = ""):


        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for

        # Optimizer-specific logic
        self._compute_adjustment_fn = compute_adjustment_fn
        self._get_state_fn = get_state_fn

        # Store state headers/operators (will combine with batch headers later)
        self._state_headers = state_headers
        self._state_operators = state_operators

        # Update headers are always the same
        self._backprop_popup_headers_single = universal_blame_headers_single
        self._backprop_popup_operators_single = universal_blame_operators_single

        self._backprop_popup_headers_batch = universal_blame_headers_batch
        self._backprop_popup_operators_batch = universal_blame_operators_batch

        # Finalizer headers will be set by configure_optimizer based on batch_size
        self._backprop_popup_headers_finalizer = None
        self._backprop_popup_operators_finalizer = None

        # Wrap the universal functions with optimizer instance
        self.update = self._intercept_update
        self.finalizer = self._intercept_finalize



    def compute_adjustment(self, neuron, weight_id, avg_blame, lr):
        """Call optimizer-specific adjustment logic"""
        return self._compute_adjustment_fn(neuron, weight_id, avg_blame, lr)

    def get_state(self, neuron, weight_id):
        """Call optimizer-specific state retrieval"""
        return self._get_state_fn(neuron, weight_id)

    def configure_optimizer(self, config):
        # NOTE THIS RUNS AFTER TRAINING!!! BAH!!!!! FIXME
        """Configure batch size - convert 0 to full batch, leave None for AutoML"""
        print("******* Strategy Optimizer Config ")
        if config.batch_size == 0:
            config.batch_size = config.training_data.sample_count
        elif config.batch_size is None:
            # AutoML will set this in Config.py
            pass
        elif isinstance(config.batch_size, int) and config.batch_size > 0:
            if config.batch_size > config.training_data.sample_count:
                print(f"Warning: batch_size ({config.batch_size}) exceeds total sample count "
                      f"({config.training_data.sample_count}). Using full batch.")
                config.batch_size = config.training_data.sample_count
        else:
            raise ValueError(
                f"batch_size must be a positive integer, 0 (full batch), or None (auto). Got: {config.batch_size}")


        # Build finalizer headers based on batch size
        if config.batch_size == 1:
            # Single-sample: Skip batch mechanics (they're redundant when count=1)
            self._backprop_popup_headers_finalizer = self._state_headers
            self._backprop_popup_operators_finalizer = self._state_operators
        else:
            # Batch mode: Show batch mechanics + optimizer state
            self._backprop_popup_headers_finalizer = universal_batch_headers + self._state_headers
            self._backprop_popup_operators_finalizer = universal_batch_operators + self._state_operators

        # Return appropriate headers
        if config.batch_size == 1:
            return (self._backprop_popup_headers_single,
                    self._backprop_popup_operators_single,
                    self._backprop_popup_headers_finalizer,
                    self._backprop_popup_operators_finalizer)
        else:
            return (self._backprop_popup_headers_batch,
                    self._backprop_popup_operators_batch,
                    self._backprop_popup_headers_finalizer,
                    self._backprop_popup_operators_finalizer)
    def _intercept_update(self, neuron, input_vector, blame, t, config, epoch, iteration, batch_id):
        """Wrapper that adds context to universal update logs"""
        raw_logs = universal_batch_update(neuron, input_vector, blame, t, config)
        ctx = [epoch, iteration, neuron.nid, batch_id]

        final_logs = []
        for row in raw_logs:
            weight_id, *rest = row
            final_logs.append([*ctx[:3], weight_id, ctx[3], *rest])

        return final_logs

    def _intercept_finalize(self, batch_size, epoch, iteration, batch_id):
        """Wrapper that adds context to universal finalize logs"""
        raw_logs = universal_batch_finalize(batch_size, self) or []

        final_logs = []
        for row in raw_logs:
            neuron_id, weight_id, *rest = row
            prefix = [epoch, iteration, neuron_id, weight_id, batch_id]
            final_logs.append(prefix + rest)

        return final_logs

    def __repr__(self):
        return self.name


# ==============================================================================
# OPTIMIZER IMPLEMENTATIONS
# ==============================================================================

def sgd_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    SGD: The simplest optimizer.
    Adjustment = learning_rate × average_blame
    """
    return lr * avg_blame


def sgd_get_state(neuron, weight_id):
    """SGD shows learning rate as part of its calculation"""
    lr = neuron.learning_rates[weight_id]
    return [lr]

Optimizer_SGD = StrategyOptimizer(
    name="Stochastic Gradient Descent",
    desc="Updates weights using the raw gradient scaled by learning rate.",
    when_to_use="Simple problems, shallow networks, or when implementing your own optimizer.",
    best_for="Manual tuning, simple models, or teaching tools.",
    compute_adjustment_fn=sgd_compute_adjustment,
    get_state_fn=sgd_get_state,
    state_headers=["Lrn Rt"],
    state_operators=[" "]
)

# ==============================================================================
# SIMPLEX OPTIMIZER (Adaptive per-weight learning rates)
# ==============================================================================

def simplex_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    Simplex: Auto-tuning learning rate per weight.
    - If adjustment would be too large (explosion risk): halt update and cut LR in half
    - If adjustment is reasonable (stable): grow LR by 5%

    This creates per-weight adaptive learning that finds optimal rates automatically.
    """
    too_high_threshold =  Optimizer_Simplex.config.training_data.input_max * 5  # Explosion detection threshold
    decay_rate = 0.5  # LR multiplier when explosion detected
    growth_rate = 1.05  # LR multiplier when stable

    # Calculate raw adjustment
    raw_adjustment = avg_blame * lr

    # Explosion detection
    if abs(raw_adjustment) > too_high_threshold:
        # Too aggressive - halt this update and reduce learning rate
        adjustment = 0.0
        neuron.learning_rates[weight_id] *= decay_rate
    else:
        # Stable - use the adjustment and grow learning rate
        adjustment = raw_adjustment
        neuron.learning_rates[weight_id] *= growth_rate

    return adjustment


def simplex_get_state(neuron, weight_id):
    """
    Simplex state: Shows learning rate and whether explosion occurred
    """
    too_high_threshold = 10.0

    lr = neuron.learning_rates[weight_id]

    # Determine what happened this step (for logging)
    # We can infer from the LR whether it grew or shrank
    # Note: This is called AFTER compute_adjustment modified the LR
    # So we're showing the NEW lr after adjustment

    return [
        lr,  # Current learning rate (after adjustment)
    ]


Optimizer_Simplex = StrategyOptimizer(
    name="Simplex",
    desc="Auto-tuning per-weight learning rates - grows when stable, shrinks when exploding.",
    when_to_use="When you want automatic learning rate tuning without manual scheduling.",
    best_for="General purpose; automatically finds optimal per-weight learning rates.",
    compute_adjustment_fn=simplex_compute_adjustment,
    get_state_fn=simplex_get_state,
    state_headers=["Lrn Rt"],
    state_operators=[" "]
)


def rmsprop_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    RMSprop: Scales learning rate by moving average of squared gradients.
    Helps handle varying gradient magnitudes across different weights.
    """
    beta = 0.9  # Decay rate for moving average
    epsilon = 1e-8  # Small constant to avoid division by zero

    # Update moving average of squared gradients
    neuron.v[weight_id] = beta * neuron.v[weight_id] + (1 - beta) * (avg_blame ** 2)

    # Scale adjustment by square root of moving average
    sqrt_v = neuron.v[weight_id] ** 0.5
    adjustment = lr * avg_blame / (sqrt_v + epsilon)

    return adjustment


def rmsprop_get_state1(neuron, weight_id):
    """
    RMSprop state: v (moving avg of squared gradients) and sqrt(v)
    """
    return [
        neuron.v[weight_id],
        neuron.v[weight_id] ** 0.5
    ]


def rmsprop_get_state(neuron, weight_id):
    """RMSprop state with intermediate calculation"""
    epsilon = 1e-8
    sqrt_v = neuron.v[weight_id] ** 0.5
    lr = neuron.learning_rates[weight_id]
    scaled_lr = lr / (sqrt_v + epsilon)  # Show the scaling!

    return [
        neuron.v[weight_id],
        sqrt_v,
        scaled_lr  # NEW: The scaled learning rate
    ]


Optimizer_RMSprop = StrategyOptimizer(
    name="RMSprop",
    desc="Root Mean Square Propagation - scales learning rate by moving average of squared gradients.",
    when_to_use="Good for RNNs and non-stationary objectives; handles noisy gradients well.",
    best_for="Recurrent networks, time-series problems, or when gradients vary widely.",
    compute_adjustment_fn=rmsprop_compute_adjustment,
    get_state_fn=rmsprop_get_state,
    state_headers=["v", "sqrt(v)", "Scaled LR"],
    state_operators=[" ", "÷", " "]  # Shows division operator
)


# ==============================================================================
# MOMENTUM OPTIMIZER
# ==============================================================================

def momentum_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    Momentum: Accumulates velocity (exponential moving average of gradients).
    Helps accelerate in consistent directions and dampen oscillations.
    """
    beta = 0.9  # Momentum coefficient

    # Update velocity (exponential moving average of gradients)
    neuron.m[weight_id] = beta * neuron.m[weight_id] + (1 - beta) * avg_blame

    # Apply velocity to learning rate
    adjustment = lr * neuron.m[weight_id]

    return adjustment


def momentum_get_state(neuron, weight_id):
    """
    Momentum state: velocity and scaled learning rate
    """
    lr = neuron.learning_rates[weight_id]
    scaled_lr = lr * neuron.m[weight_id]

    return [
        neuron.m[weight_id],  # velocity
        scaled_lr  # effective adjustment rate
    ]


Optimizer_Momentum = StrategyOptimizer(
    name="Momentum",
    desc="SGD with momentum - accumulates velocity to accelerate learning in consistent directions.",
    when_to_use="When gradients are noisy but have consistent overall direction.",
    best_for="Deep networks, image classification, avoiding local minima.",
    compute_adjustment_fn=momentum_compute_adjustment,
    get_state_fn=momentum_get_state,
    state_headers=["velocity", "Scaled LR"],
    state_operators=[" ", " "]
)




# ==============================================================================
# ADAM OPTIMIZER
# ==============================================================================

def adam_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    Adam: Combines momentum (first moment) and RMSprop (second moment)
    with bias correction for early training steps.
    """
    beta1 = 0.9  # Momentum decay rate
    beta2 = 0.999  # RMSprop decay rate
    epsilon = 1e-8  # Numerical stability

    # Increment timestep (per neuron)
    neuron.t += 1

    # Update biased first moment (momentum)
    neuron.m[weight_id] = beta1 * neuron.m[weight_id] + (1 - beta1) * avg_blame

    # Update biased second moment (RMSprop-style variance)
    neuron.v[weight_id] = beta2 * neuron.v[weight_id] + (1 - beta2) * (avg_blame ** 2)

    # Compute bias-corrected estimates
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
    v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)

    # Compute adjustment
    adjustment = lr * m_hat / (v_hat ** 0.5 + epsilon)

    return adjustment


def adam_get_state(neuron, weight_id):
    """
    Adam state: Shows both biased and bias-corrected moments
    """
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Compute bias-corrected estimates
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
    v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)

    lr = neuron.learning_rates[weight_id]
    scaled_lr = lr / (v_hat ** 0.5 + epsilon)

    return [
        neuron.m[weight_id],  # First moment (momentum)
        neuron.v[weight_id],  # Second moment (variance)
        neuron.t,  # Timestep
        m_hat,  # Bias-corrected momentum
        v_hat,  # Bias-corrected variance
        scaled_lr  # Effective learning rate
    ]


Optimizer_Adam = StrategyOptimizer(
    name="Adam",
    desc="Adaptive Moment Estimation - combines momentum and RMSprop with bias correction.",
    when_to_use="Default choice for most deep learning tasks; handles sparse gradients well.",
    best_for="Most deep learning tasks, requires minimal tuning, widely used.",
    compute_adjustment_fn=adam_compute_adjustment,
    get_state_fn=adam_get_state,
    state_headers=["m", "v", "t", "m_hat", "v_hat", "Scaled LR"],
    state_operators=[" ", " ", " ", " ", " ", " "]
)


# ==============================================================================
# ADAGRAD OPTIMIZER
# ==============================================================================

def adagrad_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    AdaGrad: Adapts learning rate based on cumulative squared gradients.
    Learning rate decreases over time (good for sparse features).
    """
    epsilon = 1e-8

    # Accumulate squared gradients (no decay - this is key difference from RMSprop)
    neuron.v[weight_id] += avg_blame ** 2

    # Compute adjustment with decreasing effective learning rate
    adjustment = lr * avg_blame / (neuron.v[weight_id] ** 0.5 + epsilon)

    return adjustment


def adagrad_get_state(neuron, weight_id):
    """
    AdaGrad state: cumulative squared gradients
    """
    epsilon = 1e-8
    sqrt_G = neuron.v[weight_id] ** 0.5

    lr = neuron.learning_rates[weight_id]
    scaled_lr = lr / (sqrt_G + epsilon)

    return [
        neuron.v[weight_id],  # G (cumulative squared gradients)
        sqrt_G,  # sqrt(G)
        scaled_lr  # Effective learning rate (decreases over time)
    ]


Optimizer_AdaGrad = StrategyOptimizer(
    name="AdaGrad",
    desc="Adaptive Gradient - accumulates all past squared gradients (learning rate decreases over time).",
    when_to_use="Sparse features, NLP tasks, when different features need very different learning rates.",
    best_for="Sparse data, word embeddings, when features have vastly different frequencies.",
    compute_adjustment_fn=adagrad_compute_adjustment,
    get_state_fn=adagrad_get_state,
    state_headers=["G", "sqrt(G)", "Scaled LR"],
    state_operators=[" ", " ", " "]
)


# ==============================================================================
# NADAM OPTIMIZER (Adam + Nesterov Momentum)
# ==============================================================================

def nadam_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    NAdam: Adam with Nesterov momentum (look-ahead gradient).
    Slightly more aggressive updates than Adam.
    """
    beta1 = 0.9  # Momentum decay rate
    beta2 = 0.999  # RMSprop decay rate
    epsilon = 1e-8

    # Increment timestep
    neuron.t += 1

    # Update biased first moment
    neuron.m[weight_id] = beta1 * neuron.m[weight_id] + (1 - beta1) * avg_blame

    # Update biased second moment
    neuron.v[weight_id] = beta2 * neuron.v[weight_id] + (1 - beta2) * (avg_blame ** 2)

    # Compute bias-corrected estimates
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
    v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)

    # Nesterov momentum: look ahead by adding current gradient
    m_nesterov = beta1 * m_hat + (1 - beta1) * avg_blame / (1 - beta1 ** neuron.t)

    # Compute adjustment
    adjustment = lr * m_nesterov / (v_hat ** 0.5 + epsilon)

    return adjustment


def nadam_get_state(neuron, weight_id):
    """
    NAdam state: similar to Adam but shows Nesterov-corrected momentum
    """
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Compute bias-corrected estimates
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
    v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)

    # Nesterov correction
    # Note: We'd need to store avg_blame to show exact m_nesterov here
    # For display, we'll show the key components

    lr = neuron.learning_rates[weight_id]
    scaled_lr = lr / (v_hat ** 0.5 + epsilon)

    return [
        neuron.m[weight_id],  # First moment
        neuron.v[weight_id],  # Second moment
        neuron.t,  # Timestep
        m_hat,  # Bias-corrected momentum
        v_hat,  # Bias-corrected variance
        scaled_lr  # Effective learning rate
    ]


Optimizer_NAdam = StrategyOptimizer(
    name="NAdam",
    desc="Nesterov-accelerated Adam - combines Adam with Nesterov momentum for faster convergence.",
    when_to_use="When you want Adam's adaptivity with more aggressive momentum.",
    best_for="Deep networks where Adam works but you want potentially faster convergence.",
    compute_adjustment_fn=nadam_compute_adjustment,
    get_state_fn=nadam_get_state,
    state_headers=["m", "v", "t", "m_hat", "v_hat", "Scaled LR"],
    state_operators=[" ", " ", " ", " ", " ", " "]
)


# ==============================================================================
# ADAMW OPTIMIZER (Adam with Decoupled Weight Decay)
# ==============================================================================

def adamw_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    AdamW: Adam with decoupled weight decay.
    Fixes issues with L2 regularization in original Adam.
    Weight decay is applied directly to weights, not through gradients.
    """
    beta1 = 0.9  # Momentum decay rate
    beta2 = 0.999  # RMSprop decay rate
    epsilon = 1e-8
    weight_decay = 0.01  # Decoupled weight decay coefficient

    # Increment timestep
    neuron.t += 1

    # Update biased first moment (momentum)
    neuron.m[weight_id] = beta1 * neuron.m[weight_id] + (1 - beta1) * avg_blame

    # Update biased second moment (variance)
    neuron.v[weight_id] = beta2 * neuron.v[weight_id] + (1 - beta2) * (avg_blame ** 2)

    # Compute bias-corrected estimates
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
    v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)

    # Adam adjustment
    adam_adjustment = lr * m_hat / (v_hat ** 0.5 + epsilon)

    # Add decoupled weight decay (applied to weight, not gradient!)
    if weight_id == 0:
        current_weight = neuron.bias
    else:
        current_weight = neuron.weights[weight_id - 1]

    weight_decay_adjustment = lr * weight_decay * current_weight

    # Total adjustment = Adam part + weight decay part
    adjustment = adam_adjustment + weight_decay_adjustment

    return adjustment


def adamw_get_state(neuron, weight_id):
    """
    AdamW state: Shows Adam components plus weight decay contribution
    """
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.01

    # Compute bias-corrected estimates
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
    v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)

    lr = neuron.learning_rates[weight_id]
    scaled_lr = lr / (v_hat ** 0.5 + epsilon)

    # Current weight for decay calculation
    if weight_id == 0:
        current_weight = neuron.bias
    else:
        current_weight = neuron.weights[weight_id - 1]

    wd_contribution = lr * weight_decay * current_weight

    return [
        neuron.m[weight_id],  # First moment
        neuron.v[weight_id],  # Second moment
        m_hat,  # Bias-corrected momentum
        v_hat,  # Bias-corrected variance
        scaled_lr,  # Effective learning rate
        wd_contribution  # Weight decay contribution
    ]


Optimizer_AdamW = StrategyOptimizer(
    name="AdamW",
    desc="Adam with decoupled Weight decay - fixes L2 regularization issues in Adam.",
    when_to_use="When you need regularization with Adam; standard for transformers and modern NLP.",
    best_for="Large models, transformers, when you need both adaptive learning and regularization.",
    compute_adjustment_fn=adamw_compute_adjustment,
    get_state_fn=adamw_get_state,
    state_headers=["m", "v", "m_hat", "v_hat", "Scaled LR", "WD"],
    state_operators=[" ", " ", " ", " ", " ", "+"]
)


# ==============================================================================
# ADADELTA OPTIMIZER (AdaGrad without learning rate!)
# ==============================================================================

def adadelta_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    Adadelta: Extension of AdaGrad that doesn't require manual learning rate.
    Uses moving average of squared gradients AND squared updates.
    The 'lr' parameter is ignored - Adadelta is learning-rate-free!
    """
    rho = 0.95  # Decay rate for moving averages
    epsilon = 1e-6  # Numerical stability

    # Initialize delta accumulator if needed (reuse 'm' for this)
    # m stores accumulated squared updates
    # v stores accumulated squared gradients

    # Update accumulated squared gradient
    neuron.v[weight_id] = rho * neuron.v[weight_id] + (1 - rho) * (avg_blame ** 2)

    # Compute RMS of previous updates and current gradients
    rms_delta = (neuron.m[weight_id] + epsilon) ** 0.5
    rms_grad = (neuron.v[weight_id] + epsilon) ** 0.5

    # Compute adjustment (note: no learning rate!)
    adjustment = (rms_delta / rms_grad) * avg_blame

    # Update accumulated squared updates
    neuron.m[weight_id] = rho * neuron.m[weight_id] + (1 - rho) * (adjustment ** 2)

    return adjustment


def adadelta_get_state(neuron, weight_id):
    """
    Adadelta state: Shows the adaptive learning rate it computes
    """
    epsilon = 1e-6

    rms_delta = (neuron.m[weight_id] + epsilon) ** 0.5
    rms_grad = (neuron.v[weight_id] + epsilon) ** 0.5

    # The "learning rate" Adadelta computes
    adaptive_lr = rms_delta / rms_grad

    return [
        neuron.v[weight_id],  # Accumulated squared gradients
        neuron.m[weight_id],  # Accumulated squared updates
        rms_grad,  # RMS of gradients
        rms_delta,  # RMS of updates
        adaptive_lr  # Computed learning rate
    ]


Optimizer_Adadelta = StrategyOptimizer(
    name="Adadelta",
    desc="Extension of AdaGrad that doesn't require manual learning rate - computes it automatically.",
    when_to_use="When you want adaptive learning without tuning learning rate; no LR needed!",
    best_for="When you want 'set and forget' training, RNNs, speech recognition.",
    compute_adjustment_fn=adadelta_compute_adjustment,
    get_state_fn=adadelta_get_state,
    state_headers=["Grad²", "Δ²", "RMS(g)", "RMS(Δ)", "Adaptive LR"],
    state_operators=[" ", " ", " ", "÷", " "]
)


# ==============================================================================
# ADAMAX OPTIMIZER (Adam variant using infinity norm)
# ==============================================================================

def adamax_compute_adjustment(neuron, weight_id, avg_blame, lr):
    """
    AdaMax: Variant of Adam based on infinity norm.
    More stable than Adam for some problems, especially with sparse gradients.
    """
    beta1 = 0.9  # Momentum decay rate
    beta2 = 0.999  # For infinity norm estimation
    epsilon = 1e-8

    # Increment timestep
    neuron.t += 1

    # Update biased first moment (momentum)
    neuron.m[weight_id] = beta1 * neuron.m[weight_id] + (1 - beta1) * avg_blame

    # Update infinity norm estimate (exponentially weighted max)
    # This is the key difference from Adam - max instead of squared average
    neuron.v[weight_id] = max(beta2 * neuron.v[weight_id], abs(avg_blame))

    # Compute bias-corrected momentum
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)

    # Compute adjustment (note: v doesn't need bias correction for infinity norm)
    adjustment = lr * m_hat / (neuron.v[weight_id] + epsilon)

    return adjustment


def adamax_get_state(neuron, weight_id):
    """
    AdaMax state: Shows momentum and infinity norm
    """
    beta1 = 0.9
    epsilon = 1e-8

    # Compute bias-corrected momentum
    m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)

    lr = neuron.learning_rates[weight_id]
    scaled_lr = lr / (neuron.v[weight_id] + epsilon)

    return [
        neuron.m[weight_id],  # First moment (momentum)
        neuron.v[weight_id],  # Infinity norm estimate
        neuron.t,  # Timestep
        m_hat,  # Bias-corrected momentum
        scaled_lr  # Effective learning rate
    ]


Optimizer_AdaMax = StrategyOptimizer(
    name="AdaMax",
    desc="Adam variant using infinity norm - more stable for sparse gradients than Adam.",
    when_to_use="When Adam is unstable; good for embeddings and sparse features.",
    best_for="NLP, sparse data, when Adam diverges or is unstable.",
    compute_adjustment_fn=adamax_compute_adjustment,
    get_state_fn=adamax_get_state,
    state_headers=["m", "u_∞", "t", "m_hat", "Scaled LR"],
    state_operators=[" ", " ", " ", " ", " "]
)

# ==============================================================================
# MAGDETOX 2.0 OPTIMIZER (Dynamic Directional Step)
# ==============================================================================

# Define a base hyperparameter for the optimizer itself
ALPHA_BASE = 0.01


def magdetox_compute_adjustment(neuron, weight_id, avg_blame, blame_magnitude):
    """
    MagDetox 2.0:

    1. Breaks Magnitude Bullying by taking the sign of the corrupted avg_blame.
    2. Breaks Dead-Stop/Overshoot by using 'blame_magnitude' (the Accepted Blame |delta|)
       to create a dynamic step size.

    The 'blame_magnitude' parameter is ASSUMED to be the Average Accepted Blame (|delta|).
    """

    # 1. Get the direction from the CORRUPTED signal (avg_blame).
    # This ignores the magnitude of the input (a_i) and fixes Magnitude Bullying.
    if abs(avg_blame) < 1e-15:
        return 0.0

    sign_of_blame = 1.0 if avg_blame > 0 else -1.0

    # 2. Calculate the DYNAMIC STEP MAGNITUDE.
    # The step size scales with the uncorrupted error magnitude (blame_factor).
    # This solves the dead-stop (large |delta| -> large step) and
    # overshooting (small |delta| -> small step) problems.
    dynamic_alpha = ALPHA_BASE * blame_magnitude

    # 3. Final Adjustment
    adjustment = sign_of_blame * dynamic_alpha

    return adjustment


def magdetox_get_state(neuron, weight_id):
    """
    MagDetox state: Shows the fixed base step size (ALPHA_BASE).
    """
    return [
        ALPHA_BASE
    ]


Optimizer_MagDetox = StrategyOptimizer(
    name="MagDetox",
    desc="Applies a dynamic directional step size based on pure error magnitude (|delta|) to mitigate Magnitude Bullying and stabilize convergence.",
    when_to_use="High-magnitude regression problems with unscaled inputs.",
    best_for="Auditing the effect of input magnitude on gradient descent.",
    compute_adjustment_fn=magdetox_compute_adjustment,
    get_state_fn=magdetox_get_state,
    state_headers=["Fixed Base α"],
    state_operators=[" "]
)