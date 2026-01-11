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
        TRI.vcr_nna                     . record_weight_update(final_dict)

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
    # Optional for Adam -> state_per_weight=["m", "v"],
    # Optional for Adam -> state_per_neuron=["t"],
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
def my_custom_optimizer_calc(neuron, weight_id, TRI, avg_leverage):
    # print(f"Timestep: {TRI.timestep}")
    # print(f"Avg Leverage: {avg_leverage}")
    # print(f"LR: {neuron.learning_rates[weight_id]}")
    ...

