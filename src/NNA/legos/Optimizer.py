import math

from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.RecordSample import RecordSample
from typing import TYPE_CHECKING
if TYPE_CHECKING:     from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from src.NNA.utils.enums import RecordLevel


class StrategyOptimizer:
    """
    Represents an optimization algorithm.
    """
    def __init__(self,
        name                    : str,
        desc                    : str,
        when_to_use             : str,
        best_for                : str,
        fn_popup_info,
        fn_adj_calc,
        state_per_weight        = None,
        popup_formula           : str = None
        ):
        self.name               = name
        self.desc               = desc
        self.when               = when_to_use
        self.best               = best_for
        self.fn_popup_info      = fn_popup_info
        self.fn_adj_calc        = fn_adj_calc
        self.state_per_weight   = state_per_weight or []
        self.popup_formula      = popup_formula

    def optimize_sample(self, sample: RecordSample, TRI):
        """ Loop through each layer than neuron than weight"""
        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id in range(len(neuron.weights)): self.optimize_weight(neuron, weight_id, sample, TRI)

    def optimize_weight(self, neuron, weight_id: int, sample: RecordSample, TRI):
        """Calculate leverage from blame accumulated in the backprop procedures"""
        self.update_timestep_in_TRI       (sample, TRI)
        self.ensure_optimizer_state       (neuron)
        popup_dict                      = self.fn_popup_info(neuron, weight_id, TRI)                      # Get display values from leverage function
        popup_dict["Adj"]               = 0.0  # Assume no update until prove otherwise.
        leverage_details                = self.gather_leverage_details(neuron, weight_id)
        batch_details                   = self.gather_batch_details(neuron, weight_id, sample, TRI)
        self.check_for_adjustment       ( neuron, weight_id, sample, TRI, popup_dict)  # Delegate boundary check and recording
        final_dict                      = self.add_fields_to_dict  (neuron, weight_id, sample, TRI , popup_dict, leverage_details, batch_details)
        TRI.vcr_nna                     . record_optimizer_logic(final_dict)

    def gather_leverage_details(self, neuron, weight_id):
        """Calculate per-sample leverage details once, return dict"""
        input_value                     = neuron.neuron_inputs[weight_id]
        blame                           = neuron.accepted_blame
        leverage                        = input_value * blame
        return {
            "Input": input_value,
            "Blame": blame,
            "Leverage": leverage
        }

    def gather_batch_details(self, neuron, weight_id, sample, TRI):
        """Calculate batch statistics once, return dict"""
        if TRI.config.batch_size == 1:
            return {}  # No batch details needed

        batch_step = self.batch_step(sample.sample_id, TRI.config.batch_size)
        batch_size = self.actual_batch_size(sample.sample_id, TRI.config.batch_size, TRI.training_data.sample_count)
        accumulated = neuron.accumulated_leverage[weight_id]
        average = self.avg_leverage(accumulated, batch_step)

        return {
            "Progress": f"{batch_step}/{batch_size}",
            "Cumulative": accumulated,
            "Average": average
        }

    def check_for_adjustment(self, neuron, weight_id: int, sample: RecordSample, TRI: "TrainingRunInfo", popup_dict: dict):
        if self.is_end_of_batch(sample.sample_id, TRI):
            batch_step              = self.batch_step(sample.sample_id, TRI.config.batch_size)
            accumulated_leverage    = neuron.accumulated_leverage[weight_id]
            avg_leverage            = self.avg_leverage(accumulated_leverage, batch_step)
            adjustment              = self.fn_adj_calc(neuron, weight_id, TRI, avg_leverage)
            popup_dict["Adj"]       = adjustment
            self.update_weight        (neuron, weight_id, adjustment)

    def update_weight(self, neuron, weight_id: int, adjustment: float):
        """Perform update and reset"""
        neuron.weights[weight_id]               -= adjustment #we're getting radical and doing the intuitive add the adjustment.
        neuron.accumulated_leverage[weight_id]   = 0.0

    def batch_step(self, sample_id, batch_size):
        """Position in current batch (1-based)"""
        return ((sample_id - 1) % batch_size) + 1

    def actual_batch_size(self, sample_id, batch_size, total_samples):
        """Actual size of current batch (handles partial at end)"""
        samples_remaining = total_samples - (sample_id - 1)
        return min(batch_size, samples_remaining)

    def avg_leverage(self, accumulated, batch_step):
        """Average leverage so far in batch"""
        return accumulated / batch_step

    def update_timestep_in_TRI(self, sample, TRI):
        """Calculate and set TRI.timestep for this sample"""
        batches_per_epoch       = (TRI.training_data.sample_count + TRI.config.batch_size - 1) // TRI.config.batch_size
        batches_in_prior_epochs = (sample.epoch - 1) * batches_per_epoch
        batch_in_current_epoch  = ((sample.sample_id - 1) // TRI.config.batch_size) + 1
        TRI.timestep            = batches_in_prior_epochs + batch_in_current_epoch

    def is_end_of_batch(self, sample_id: int, TRI )->bool:
        batch_size          = TRI.config.batch_size
        if sample_id        % batch_size == 0:                  return  True #Note sample_id is 1 based NOT ZERO
        if sample_id        ==TRI.training_data.sample_count:   return True
        return False

    def add_fields_to_dict(self, neuron, weight_id: int, sample: RecordSample, TRI: "TrainingRunInfo", popup_dict: dict, leverage_details: dict, batch_details: dict) -> dict:
        """Add standard keys, batch stats (if needed), and LR/Adj at end. Return complete row for recording."""
        adjustment                  = popup_dict.pop("Adj")

        # Build in correct order:   leverage → batch → optimizer-specific → LR/Adj
        ordered_dict                        = {}
        ordered_dict.update                 (leverage_details)          # Input, Blame, Leverage
        ordered_dict.update                 (batch_details)             # Progress, Cumulative, Average (if batch)
        ordered_dict.update                 (popup_dict)                # Optimizer-specific (fn_popup_info)
        ordered_dict["LR"]                  = neuron.learning_rates[weight_id]
        ordered_dict["Adj"]                 = adjustment

        if TRI.backprop_headers is None:    TRI.backprop_headers = list(ordered_dict.keys())
        return self.inject_keys_first       (ordered_dict, run_id=TRI.run_id, epoch=sample.epoch, sample_id=sample.sample_id, nid=neuron.nid, weight_id=weight_id)

    def inject_keys_first(self, row: dict, **keys_first) -> dict:
        """
        Returns a NEW dict where:
          1) keys in `keys_first` appear first (in the order provided),
          2) then remaining items from `row` are mapped to arg_1, arg_2, etc.
        """
        out = dict(keys_first)
        for i, v in enumerate(row.values(), start=1): out[f"arg_{i}"] = v
        return out
    # ==============================================================================
    # OPTIMIZER Specific Parameters
    # ==============================================================================

    def ensure_optimizer_state(self, neuron):
        """Create optimizer_state dict and initialize arrays if needed"""
        # Called extra times but leaves clean clode intact and just does nothing on subsequent calls
        if not hasattr(neuron, 'optimizer_state'):neuron.optimizer_state = {} # First time? Create the dict
        for state_name in self.state_per_weight:                               # For each state variable this optimizer needs
            if state_name not in neuron.optimizer_state:                       # If it doesn't exist yet, create it
                neuron.optimizer_state[state_name] = [0.0] * len(neuron.weights)

# ==============================================================================
# OPTIMIZER IMPLEMENTATIONS
# ==============================================================================


def sgd_popup_info(neuron, weight_id, TRI):
    """Calculate leverage. Return display values."""

    return {
        #"quickbrownfox": "9.9498744",
        #"timestep": TRI.timestep
    }

def sgd_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """Calculate adjustment from accumulated average leverage."""
    lr = neuron.learning_rates[weight_id]
    return lr * avg_leverage

Optimizer_SGD = StrategyOptimizer(
    name="Stochastic Gradient Descent",
    desc="Updates weights using the raw gradient scaled by learning rate.",
    when_to_use="Simple problems, shallow networks, or when implementing your own optimizer.",
    best_for="Manual tuning, simple models, or teaching tools.",
    fn_popup_info=sgd_popup_info,
    fn_adj_calc=sgd_calculate_adjustment,
)


def adam_popup_info(neuron, weight_id, TRI):
    """Calculate Adam state for display. Return display values."""
    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state
    timestep    = TRI.timestep                              # Get current state
    beta1       = 0.9                                       # Adam hyperparameters
    beta2       = 0.999                                     # Adam hyperparameters
    epsilon     = 1e-8                                      # Adam hyperparameters

    # Bias correction
    m_hat       = m / (1 - beta1 ** timestep) if timestep > 0 else 0.0
    v_hat       = v / (1 - beta2 ** timestep) if timestep > 0 else 0.0

    return {
        "m": m,
        "v": v,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "timestep": timestep,
    }


def adam_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """Calculate Adam adjustment from accumulated average leverage."""

    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state
    timestep    = TRI.timestep                              # Get current state
    beta1       = 0.9                                       # Adam hyperparameters
    beta2       = 0.999                                     # Adam hyperparameters
    epsilon     = 1e-8                                      # Adam hyperparameters

    # Update momentum and velocity
    m = beta1 * m + (1 - beta1) * avg_leverage
    v = beta2 * v + (1 - beta2) * (avg_leverage ** 2)

    # Save updated state
    neuron.optimizer_state['m'][weight_id] = m
    neuron.optimizer_state['v'][weight_id] = v

    # Bias correction
    m_hat = m / (1 - beta1 ** timestep) if timestep > 0 else 0.0
    v_hat = v / (1 - beta2 ** timestep) if timestep > 0 else 0.0

    # Calculate adjustment
    lr = neuron.learning_rates[weight_id]
    adjustment = lr * m_hat / (math.sqrt(v_hat) + epsilon)

    return adjustment
Optimizer_Adam = StrategyOptimizer(
    name="Adam",
    desc="Adaptive learning rate optimizer combining momentum and RMSprop.",
    when_to_use="Most modern deep learning tasks, especially with large datasets.",
    best_for="General purpose optimization, handles sparse gradients well.",
    fn_popup_info=adam_popup_info,
    fn_adj_calc=adam_calculate_adjustment,
    state_per_weight=["m", "v"],  # ← Adam needs momentum and velocity per weight
    popup_formula="3"
    #m: Exponential moving average of the gradient (smoothed direction / “momentum” term).
    #v: Exponential moving average of the squared gradient (smoothed magnitude / scaling term).
    #m_hat: Bias-corrected m (corrects the early underestimate from starting at zero).
    #v_hat: Bias-corrected v (corrects the early underestimate from starting at zero).
)



def adam_nohat_popup_info(neuron, weight_id, TRI):
    """Calculate Adam state WITHOUT bias correction for display."""
    m = neuron.optimizer_state['m'][weight_id]
    v = neuron.optimizer_state['v'][weight_id]
    timestep = TRI.timestep

    # NO bias correction - just show raw values
    return {
        "m": m,
        "v": v,
        "timestep": timestep,
    }


def adam_nohat_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """Calculate Adam adjustment WITHOUT bias correction."""

    # Get current state
    m = neuron.optimizer_state['m'][weight_id]
    v = neuron.optimizer_state['v'][weight_id]

    # Adam hyperparameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Update momentum and velocity
    m = beta1 * m + (1 - beta1) * avg_leverage
    v = beta2 * v + (1 - beta2) * (avg_leverage ** 2)

    # Save updated state
    neuron.optimizer_state['m'][weight_id] = m
    neuron.optimizer_state['v'][weight_id] = v

    # NO BIAS CORRECTION - use raw m and v
    lr = neuron.learning_rates[weight_id]
    adjustment = lr * m / (math.sqrt(v) + epsilon)

    return adjustment


Optimizer_Adam_NoHat = StrategyOptimizer(
    name="Coverless Adam",
    desc="Adam optimizer WITHOUT bias correction - let's see if the hats matter.",
    when_to_use="Testing whether bias correction is actually needed.",
    best_for="Empirical validation over academic assumptions.",
    fn_popup_info=adam_nohat_popup_info,
    fn_adj_calc=adam_nohat_calculate_adjustment,
    state_per_weight=["m", "v"],
    popup_formula=" test2"
)


def nadam_popup_info(neuron, weight_id, TRI):
    """Nadam state display including the actual Look-Ahead value."""
    m = neuron.optimizer_state['m'][weight_id]
    v = neuron.optimizer_state['v'][weight_id]
    t = TRI.timestep
    b1, b2 = 0.9, 0.999

    # We need the last gradient (avg_leverage) to show what the look-ahead WAS.
    # If your system doesn't store last_g, we show the biased-corrected m_hat.
    m_hat = m / (1 - b1 ** t) if t > 0 else 0
    v_hat = v / (1 - b2 ** t) if t > 0 else 0

    # This is the 'Nesterov' magic value used in the last update
    # Note: In a real run, this uses the gradient from the batch that just finished.
    # For the UI, we'll label it 'm_lookahead'
    return {
        "m": m,
        "v": v,
        "m_hat": m_hat,
        "m_lookahead": b1 * m_hat,  # Simplified look-ahead for the display
        "v_hat": v_hat,
        "timestep": t
    }


def nadam_popup2_info(neuron, weight_id, TRI):
    """Retrieves the exact state used in the last batch update."""
    # Since these are in 'state_per_weight', they are persisted
    m = neuron.optimizer_state['m'][weight_id]
    v = neuron.optimizer_state['v'][weight_id]
    m_look = neuron.optimizer_state['m_lookahead'][weight_id]
    t = TRI.timestep

    return {
        "m": m,
        "v": v,
        "m_lookahead": m_look,  # No guessing!
        "timestep": t
    }


def nadam_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """Calculates adjustment and persists the look-ahead state."""
    m = neuron.optimizer_state['m'][weight_id]
    v = neuron.optimizer_state['v'][weight_id]
    t = TRI.timestep
    b1, b2, eps = 0.9, 0.999, 1e-8

    # 1. Standard Adam state updates
    m = b1 * m + (1 - b1) * avg_leverage
    v = b2 * v + (1 - b2) * (avg_leverage ** 2)

    # 2. Bias Correction
    m_hat = m / (1 - b1 ** t) if t > 0 else 0
    v_hat = v / (1 - b2 ** t) if t > 0 else 0
    g_hat = avg_leverage / (1 - b1 ** t) if t > 0 else 0

    # 3. Nadam Look-Ahead Logic
    # We blend the current momentum with the current gradient to 'peek' ahead
    m_lookahead = (b1 * m_hat) + ((1 - b1) * g_hat)

    # 4. Save EVERY piece of the puzzle
    neuron.optimizer_state['m'][weight_id] = m
    neuron.optimizer_state['v'][weight_id] = v
    neuron.optimizer_state['m_lookahead'][weight_id] = m_lookahead

    lr = neuron.learning_rates[weight_id]
    return lr * m_lookahead / (math.sqrt(v_hat) + eps)


Optimizer_Nadam = StrategyOptimizer(
    name="Nadam",
    desc="Adam with Nesterov Momentum. It peeks at the gradient to move faster.",
    when_to_use="High-dimensional spaces where you want to minimize 'lag' in momentum.",
    best_for="Fast convergence in complex architectures.",
    fn_popup_info=nadam_popup_info,
    fn_adj_calc=nadam_calculate_adjustment,
    state_per_weight=["m", "v", "m_lookahead"],
    popup_formula="test1"
)

# ==============================================================================
# ADAM OPTIMIZER
# ==============================================================================



def rmsprop_popup_info(neuron, weight_id, TRI):
    """Calculate RMSprop state for display. Return display values."""
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state
    timestep    = TRI.timestep                              # Get current state
    epsilon     = 1e-8                                      # RMSprop hyperparameters

    sqrt_v      = math.sqrt(v)
    lr          = neuron.learning_rates[weight_id]
    scaled_lr   = lr / (sqrt_v + epsilon)                   # Show the scaling

    return {
        "v": v,
        "sqrt(v)": sqrt_v,
        "Scaled LR": scaled_lr,
        "timestep": timestep,
    }


def rmsprop_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """RMSprop: Scales learning rate by moving average of squared gradients."""

    v           = neuron.optimizer_state['v'][weight_id]    # Get current state
    beta        = 0.9                                       # RMSprop hyperparameters
    epsilon     = 1e-8                                      # RMSprop hyperparameters

    # Update moving average of squared gradients
    v = beta * v + (1 - beta) * (avg_leverage ** 2)

    # Save updated state
    neuron.optimizer_state['v'][weight_id] = v

    # Calculate adjustment
    lr          = neuron.learning_rates[weight_id]
    adjustment  = lr * avg_leverage / (math.sqrt(v) + epsilon)

    return adjustment


Optimizer_RMSprop = StrategyOptimizer(
    name="RMSprop",
    desc="Root Mean Square Propagation - scales learning rate by moving average of squared gradients.",
    when_to_use="Good for RNNs and non-stationary objectives; handles noisy gradients well.",
    best_for="Recurrent networks, time-series problems, or when gradients vary widely.",
    fn_popup_info=rmsprop_popup_info,
    fn_adj_calc=rmsprop_calculate_adjustment,
    state_per_weight=["v"],  # ← RMSprop needs only v per weight
)




def momentum_popup_info(neuron, weight_id, TRI):
    """Calculate Momentum state for display. Return display values."""
    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    timestep    = TRI.timestep                              # Get current state

    lr          = neuron.learning_rates[weight_id]
    scaled_lr   = lr * m                                    # effective adjustment rate

    return {
        "velocity": m,
        "Scaled LR": scaled_lr,
        "timestep": timestep,
    }


def momentum_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """
    Momentum: Accumulates velocity (exponential moving average of gradients).
    Helps accelerate in consistent directions and dampen oscillations.
    """

    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    beta        = 0.9                                       # Momentum hyperparameters

    # Update velocity (exponential moving average of gradients)
    m = beta * m + (1 - beta) * avg_leverage

    # Save updated state
    neuron.optimizer_state['m'][weight_id] = m

    # Apply velocity to learning rate
    lr          = neuron.learning_rates[weight_id]
    adjustment  = lr * m

    return adjustment


Optimizer_Momentum = StrategyOptimizer(
    name="Momentum",
    desc="SGD with momentum - accumulates velocity to accelerate learning in consistent directions.",
    when_to_use="When gradients are noisy but have consistent overall direction.",
    best_for="Deep networks, image classification, avoiding local minima.",
    fn_popup_info=momentum_popup_info,
    fn_adj_calc=momentum_calculate_adjustment,
    state_per_weight=["m"],  # ← Momentum needs velocity per weight
)

# ==============================================================================
# ADAGRAD OPTIMIZER
# ==============================================================================
def adagrad_popup_info(neuron, weight_id, TRI):
    """Calculate AdaGrad state for display. Return display values."""
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state (G)
    timestep    = TRI.timestep                              # Get current state
    epsilon     = 1e-8                                      # AdaGrad hyperparameters

    sqrt_G      = math.sqrt(v)
    lr          = neuron.learning_rates[weight_id]
    scaled_lr   = lr / (sqrt_G + epsilon)                   # Effective learning rate (decreases over time)

    return {
        "G": v,
        "sqrt(G)": sqrt_G,
        "Scaled LR": scaled_lr,
        "timestep": timestep,
    }


def adagrad_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """
    AdaGrad: Adapts learning rate based on cumulative squared gradients.
    Learning rate decreases over time (good for sparse features).
    """

    v           = neuron.optimizer_state['v'][weight_id]    # Get current state (G)
    epsilon     = 1e-8                                      # AdaGrad hyperparameters

    # Accumulate squared gradients (no decay - this is key difference from RMSprop)
    v += avg_leverage ** 2

    # Save updated state
    neuron.optimizer_state['v'][weight_id] = v

    # Compute adjustment with decreasing effective learning rate
    lr          = neuron.learning_rates[weight_id]
    adjustment  = lr * avg_leverage / (math.sqrt(v) + epsilon)

    return adjustment


Optimizer_AdaGrad = StrategyOptimizer(
    name="AdaGrad",
    desc="Adaptive Gradient - accumulates all past squared gradients (learning rate decreases over time).",
    when_to_use="Sparse features, NLP tasks, when different features need very different learning rates.",
    best_for="Sparse data, word embeddings, when features have vastly different frequencies.",
    fn_popup_info=adagrad_popup_info,
    fn_adj_calc=adagrad_calculate_adjustment,
    state_per_weight=["v"],  # ← AdaGrad needs G per weight
)



# ==============================================================================
# ADAMW OPTIMIZER (Adam with Decoupled Weight Decay)
# ==============================================================================



def adamw_popup_info(neuron, weight_id, TRI):
    """Calculate AdamW state for display. Return display values."""
    m               = neuron.optimizer_state['m'][weight_id]    # Get current state
    v               = neuron.optimizer_state['v'][weight_id]    # Get current state
    timestep        = TRI.timestep                              # Get current state
    beta1           = 0.9                                       # AdamW hyperparameters
    beta2           = 0.999                                     # AdamW hyperparameters
    epsilon         = 1e-8                                      # AdamW hyperparameters
    weight_decay    = 0.01                                      # AdamW hyperparameters

    # Bias correction
    m_hat           = m / (1 - beta1 ** timestep) if timestep > 0 else 0.0
    v_hat           = v / (1 - beta2 ** timestep) if timestep > 0 else 0.0

    lr              = neuron.learning_rates[weight_id]
    scaled_lr       = lr / (math.sqrt(v_hat) + epsilon) if timestep > 0 else 0.0

    # Current weight for decay calculation (matches your original logic)

    current_weight = neuron.weights[weight_id ]

    wd_contribution = lr * weight_decay * current_weight

    return {
        "m": m,
        "v": v,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "Scaled LR": scaled_lr,
        "WD": wd_contribution,
        "timestep": timestep,
    }


def adamw_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """
    AdamW: Adam with decoupled weight decay.
    Fixes issues with L2 regularization in original Adam.
    Weight decay is applied directly to weights, not through gradients.
    """

    m               = neuron.optimizer_state['m'][weight_id]    # Get current state
    v               = neuron.optimizer_state['v'][weight_id]    # Get current state
    timestep        = TRI.timestep                              # Get current state
    beta1           = 0.9                                       # AdamW hyperparameters
    beta2           = 0.999                                     # AdamW hyperparameters
    epsilon         = 1e-8                                      # AdamW hyperparameters
    weight_decay    = 0.01                                      # AdamW hyperparameters

    # Update biased first moment (momentum)
    m = beta1 * m + (1 - beta1) * avg_leverage

    # Update biased second moment (variance)
    v = beta2 * v + (1 - beta2) * (avg_leverage ** 2)

    # Save updated state
    neuron.optimizer_state['m'][weight_id] = m
    neuron.optimizer_state['v'][weight_id] = v

    # Bias correction
    m_hat = m / (1 - beta1 ** timestep) if timestep > 0 else 0.0
    v_hat = v / (1 - beta2 ** timestep) if timestep > 0 else 0.0

    # Adam adjustment
    lr              = neuron.learning_rates[weight_id]
    adam_adjustment = lr * m_hat / (math.sqrt(v_hat) + epsilon) if timestep > 0 else 0.0

    # Add decoupled weight decay (applied to weight, not gradient!)
    current_weight = neuron.weights[weight_id]

    weight_decay_adjustment = lr * weight_decay * current_weight

    # Total adjustment = Adam part + weight decay part
    adjustment = adam_adjustment + weight_decay_adjustment

    return adjustment


Optimizer_AdamW = StrategyOptimizer(
    name="AdamW",
    desc="Adam with decoupled Weight decay - fixes L2 regularization issues in Adam.",
    when_to_use="When you need regularization with Adam; standard for transformers and modern NLP.",
    best_for="Large models, transformers, when you need both adaptive learning and regularization.",
    fn_popup_info=adamw_popup_info,
    fn_adj_calc=adamw_calculate_adjustment,
    state_per_weight=["m", "v"],  # ← AdamW needs momentum and velocity per weight
)



# ==============================================================================
# ADADELTA OPTIMIZER (AdaGrad without learning rate!)
# ==============================================================================


# ==============================================================================
# ADADELTA OPTIMIZER (AdaGrad without learning rate!)
# ==============================================================================


# ==============================================================================
# ADADELTA OPTIMIZER (AdaGrad without learning rate!)
# ==============================================================================

def adadelta_popup_info(neuron, weight_id, TRI):
    """Calculate Adadelta state for display. Return display values."""
    v               = neuron.optimizer_state['v'][weight_id]    # Accumulated squared gradients
    m               = neuron.optimizer_state['m'][weight_id]    # Accumulated squared updates
    timestep        = TRI.timestep                              # Get current state
    epsilon         = 1e-6                                      # Adadelta hyperparameters

    rms_grad        = math.sqrt(v + epsilon)
    rms_delta       = math.sqrt(m + epsilon)
    adaptive_lr     = rms_delta / rms_grad

    return {
        "Grad²": v,
        "Δ²": m,
        "RMS(g)": rms_grad,
        "RMS(Δ)": rms_delta,
        "Adaptive LR": adaptive_lr,
        "timestep": timestep,
    }


def adadelta_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """
    Adadelta: Extension of AdaGrad that doesn't require manual learning rate.
    Uses moving average of squared gradients AND squared updates.
    The 'lr' parameter is ignored - Adadelta is learning-rate-free!
    """

    m               = neuron.optimizer_state['m'][weight_id]    # Accumulated squared updates
    v               = neuron.optimizer_state['v'][weight_id]    # Accumulated squared gradients
    rho             = 0.95                                      # Adadelta hyperparameters
    epsilon         = 1e-6                                      # Adadelta hyperparameters

    # Update accumulated squared gradient
    v = rho * v + (1 - rho) * (avg_leverage ** 2)

    # Compute RMS of previous updates and current gradients
    rms_delta = math.sqrt(m + epsilon)
    rms_grad  = math.sqrt(v + epsilon)

    # Compute adjustment (note: no learning rate!)
    adjustment = (rms_delta / rms_grad) * avg_leverage

    # Update accumulated squared updates
    m = rho * m + (1 - rho) * (adjustment ** 2)

    # Save updated state
    neuron.optimizer_state['v'][weight_id] = v
    neuron.optimizer_state['m'][weight_id] = m

    return adjustment


Optimizer_Adadelta = StrategyOptimizer(
    name="Adadelta",
    desc="Extension of AdaGrad that doesn't require manual learning rate - computes it automatically.",
    when_to_use="When you want adaptive learning without tuning learning rate; no LR needed!",
    best_for="When you want 'set and forget' training, RNNs, speech recognition.",
    fn_popup_info=adadelta_popup_info,
    fn_adj_calc=adadelta_calculate_adjustment,
    state_per_weight=["m", "v"],  # ← Adadelta needs squared updates and squared gradients per weight
)

# ==============================================================================
# ADAMAX OPTIMIZER (Adam variant using infinity norm)
# ==============================================================================
def adamax_popup_info(neuron, weight_id, TRI):
    """Calculate AdaMax state for display. Return display values."""
    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state (u_∞)
    timestep    = TRI.timestep                              # Get current state
    beta1       = 0.9                                       # AdaMax hyperparameters
    epsilon     = 1e-8                                      # AdaMax hyperparameters

    # Bias-corrected momentum
    m_hat       = m / (1 - beta1 ** timestep) if timestep > 0 else 0.0

    lr          = neuron.learning_rates[weight_id]
    scaled_lr   = lr / (v + epsilon)

    return {
        "m": m,
        "u_∞": v,
        "timestep": timestep,
        "m_hat": m_hat,
        "Scaled LR": scaled_lr,
    }


def adamax_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """
    AdaMax: Variant of Adam based on infinity norm.
    More stable than Adam for some problems, especially with sparse gradients.
    """

    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state (u_∞)
    timestep    = TRI.timestep                              # Get current state
    beta1       = 0.9                                       # AdaMax hyperparameters
    beta2       = 0.999                                     # AdaMax hyperparameters
    epsilon     = 1e-8                                      # AdaMax hyperparameters

    # Update biased first moment (momentum)
    m = beta1 * m + (1 - beta1) * avg_leverage

    # Update infinity norm estimate (exponentially weighted max)
    v = max(beta2 * v, abs(avg_leverage))

    # Save updated state
    neuron.optimizer_state['m'][weight_id] = m
    neuron.optimizer_state['v'][weight_id] = v

    # Compute bias-corrected momentum
    m_hat = m / (1 - beta1 ** timestep) if timestep > 0 else 0.0

    # Compute adjustment (note: v doesn't need bias correction for infinity norm)
    lr          = neuron.learning_rates[weight_id]
    adjustment  = lr * m_hat / (v + epsilon) if timestep > 0 else 0.0

    return adjustment


Optimizer_AdaMax = StrategyOptimizer(
    name="AdaMax",
    desc="Adam variant using infinity norm - more stable for sparse gradients than Adam.",
    when_to_use="When Adam is unstable; good for embeddings and sparse features.",
    best_for="NLP, sparse data, when Adam diverges or is unstable.",
    fn_popup_info=adamax_popup_info,
    fn_adj_calc=adamax_calculate_adjustment,
    state_per_weight=["m", "v"],  # ← AdaMax needs momentum and infinity norm estimate per weight
)



# ==============================================================================
# RADAM OPTIMIZER (Rectified Adam)
# ==============================================================================

def radam_popup_info(neuron, weight_id, TRI):
    """Calculate RAdam state for display. Return display values."""
    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state
    timestep    = TRI.timestep                              # Get current state
    beta1       = 0.9                                       # RAdam hyperparameters
    beta2       = 0.999                                     # RAdam hyperparameters
    epsilon     = 1e-8                                      # RAdam hyperparameters

    if timestep <= 0:
        return {
            "m": m,
            "v": v,
            "m_hat": 0.0,
            "v_hat": 0.0,
            "rho_t": 0.0,
            "r_t": 0.0,
            "Scaled LR": 0.0,
            "timestep": timestep,
        }

    # Bias correction
    m_hat       = m / (1 - beta1 ** timestep)
    v_hat       = v / (1 - beta2 ** timestep)

    # RAdam rectification
    rho_inf     = (2 / (1 - beta2)) - 1
    beta2_t     = beta2 ** timestep
    rho_t       = rho_inf - (2 * timestep * beta2_t) / (1 - beta2_t)

    if rho_t > 4:
        r_t = math.sqrt(
            ((rho_t - 4) * (rho_t - 2) * rho_inf)
            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
        )
        denom = (math.sqrt(v_hat) + epsilon)
        lr    = neuron.learning_rates[weight_id]
        scaled_lr = (lr * r_t) / denom
    else:
        r_t = 0.0
        lr  = neuron.learning_rates[weight_id]
        scaled_lr = lr  # In the unrectified regime, RAdam behaves like momentum SGD using m_hat

    return {
        "m": m,
        "v": v,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "rho_t": rho_t,
        "r_t": r_t,
        "Scaled LR": scaled_lr,
        "timestep": timestep,
    }


def radam_calculate_adjustment(neuron, weight_id, TRI, avg_leverage):
    """
    RAdam: Rectified Adam.
    Behaves like Adam, but "rectifies" the adaptive denominator early in training.
    When there isn't enough reliable variance information yet, it falls back to a momentum-style step.
    """

    m           = neuron.optimizer_state['m'][weight_id]    # Get current state
    v           = neuron.optimizer_state['v'][weight_id]    # Get current state
    timestep    = TRI.timestep                              # Get current state
    beta1       = 0.9                                       # RAdam hyperparameters
    beta2       = 0.999                                     # RAdam hyperparameters
    epsilon     = 1e-8                                      # RAdam hyperparameters

    # Update moments
    m = beta1 * m + (1 - beta1) * avg_leverage
    v = beta2 * v + (1 - beta2) * (avg_leverage ** 2)

    # Save updated state
    neuron.optimizer_state['m'][weight_id] = m
    neuron.optimizer_state['v'][weight_id] = v

    if timestep <= 0:
        return 0.0

    # Bias correction
    m_hat = m / (1 - beta1 ** timestep)

    # RAdam rectification
    rho_inf = (2 / (1 - beta2)) - 1
    beta2_t = beta2 ** timestep
    rho_t   = rho_inf - (2 * timestep * beta2_t) / (1 - beta2_t)

    lr = neuron.learning_rates[weight_id]

    if rho_t > 4:
        v_hat = v / (1 - beta2 ** timestep)
        r_t = math.sqrt(
            ((rho_t - 4) * (rho_t - 2) * rho_inf)
            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
        )
        adjustment = lr * r_t * m_hat / (math.sqrt(v_hat) + epsilon)
    else:
        # Not enough variance information yet -> momentum-style step
        adjustment = lr * m_hat

    return adjustment


Optimizer_RAdam = StrategyOptimizer(
    name="RAdam",
    desc="Rectified Adam - Adam with an early-training variance rectification to improve stability.",
    when_to_use="When Adam is a bit twitchy early or needs warmup; often 'just works' out of the box.",
    best_for="General purpose deep learning, especially when early-step stability matters.",
    fn_popup_info=radam_popup_info,
    fn_adj_calc=radam_calculate_adjustment,
    state_per_weight=["m", "v"],  # ← RAdam needs momentum and velocity per weight
)

