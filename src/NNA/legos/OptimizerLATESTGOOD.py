from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.RecordSample import RecordSample

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
        fn_adj_calc
        ):
        self.name               = name
        self.desc               = desc
        self.when               = when_to_use
        self.best               = best_for
        self.fn_popup_info      = fn_popup_info
        self.fn_adj_calc        = fn_adj_calc

        #DB Buffer
        self.weight_update_buffer   = []
        self.buffer_limit           = 5000

    def optimize_sample(self, sample: RecordSample, TRI):
        """ Loop through each layer than neuron than weight"""
        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id in range(len(neuron.weights)): self.optimize_weight(neuron, weight_id, sample, TRI)
        self.flush(TRI)

    def optimize_weight(self, neuron, weight_id: int, sample: RecordSample, TRI):
        """Calculate leverage from blame accumulated in the backprop procedures"""
        row = self.fn_popup_info(neuron, weight_id, TRI)                      # Get display values from leverage function
        self.update_weight(neuron, weight_id, sample, TRI, row)    # Delegate boundary check and recording

    def update_weight(self, neuron, weight_id: int, sample: RecordSample, TRI, row: dict):
        """Check boundary, calculate adjustment if needed, record"""

        is_boundary         = self.is_end_of_batch(sample.sample_id, TRI)

        if is_boundary:
            avg_leverage    = neuron.accumulated_leverage[weight_id] / TRI.config.batch_size
            adjustment      = self.fn_adj_calc(neuron, weight_id, TRI, avg_leverage)
            neuron          . weights[weight_id] -= adjustment
            neuron          . accumulated_leverage[weight_id] = 0.0
        else: adjustment    = 0.0

        # Add LR and Adj to row for recording
        row["LR"] = neuron.learning_rates[weight_id]
        row["Adj"] = adjustment

        # Add standard keys and record
        row = self.inject_keys_first(row, run_id=TRI.run_id, epoch=sample.epoch, sample_id=sample.sample_id, nid=neuron.nid, weight_id=weight_id)
        self.record_backprop_details(TRI, row)

    def record_backprop_details(self, TRI, record: dict):
        """Buffers weight update. Flushes when buffer reaches limit."""
        self.weight_update_buffer.append(record)
        if len(self.weight_update_buffer) >= self.buffer_limit: self.flush(TRI)

    def flush(self, TRI):
        """Write buffered weight updates to DB"""

        if not self.weight_update_buffer:            return
        if not TRI.should_record(RecordLevel.FULL):  return

        sample_row          = self.weight_update_buffer[0]
        columns             = list(sample_row.keys())
        placeholders        = ", ".join(["?"] * len(columns))
        columns_str         = ", ".join(columns)
        sql                 = f"INSERT INTO WeightAdjustments ({columns_str}) VALUES ({placeholders})"
        rows                = [tuple(row[col] for col in columns) for row in self.weight_update_buffer]
        TRI.db.executemany  ( sql, rows, "weight adjustments")
        self                . weight_update_buffer.clear()

    def is_end_of_batch(self, sample_id: int, TRI )->bool:
        batch_size          = TRI.config.batch_size
        if sample_id        % batch_size == 0:                  return  True #Note sample_id is 1 based NOT ZERO
        if sample_id        ==TRI.training_data.sample_count:   return True
        return False

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
# OPTIMIZER IMPLEMENTATIONS
# ==============================================================================


def sgd_calculate_leverage(neuron, weight_id, TRI):
    """Calculate leverage. Return display values."""
    input_value = neuron.neuron_inputs[weight_id]
    blame = neuron.accepted_blame
    leverage = input_value * blame


    return {
        "Input": input_value,
        "Blame": blame,
        "Leverage": leverage,
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
    fn_popup_info=sgd_calculate_leverage,
    fn_adj_calc=sgd_calculate_adjustment,
    # Optional for Adam -> state_per_weight=["m", "v"],
    # Optional for Adam -> state_per_neuron=["t"],
)
