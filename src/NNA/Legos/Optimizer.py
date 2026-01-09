from enum import IntEnum, auto

from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.RecordSample import RecordSample


class StrategyOptimizer:
    """
    Represents an optimization algorithm.
    """
    def __init__(self,
        name                    : str,
        desc                    : str,
        when_to_use             : str,
        best_for                : str,
        optimizer_brain,
        ):
        self.name               = name
        self.desc               = desc
        self.when               = when_to_use
        self.best               = best_for
        self.optimizer_brain    = optimizer_brain

        #DB Buffer
        self.weight_update_buffer   = []
        self.buffer_limit           = 5000

    def is_end_of_batch(self, sample: RecordSample, TRI )->bool:
        batch_size      = TRI.config.batch_size
        sample_id       = sample.sample_id #Note sample_id is 1 based NOT ZERO
        if sample_id    % batch_size == 0: return  True
        if sample_id    == TRI.training_data.sample_count: return True
        return False


    def inject_keys_first(self, row: dict, **keys_first) -> dict:
        """
        Returns a NEW dict where:
          1) "column_schema" is the very first key. It is NOT passed in; it is GENERATED from `row`
             by joining the non-key columns (in `row`'s current order) with "|".
          2) then keys in `keys_first` appear next (in the order provided),
          3) then all remaining items from `row` in their existing order.

        Values in `keys_first` override any same-named keys already in `row`.
        """
        schema_cols = [
            k for k in row.keys()
            if k != "column_schema" and k not in keys_first
        ]
        out = {"column_schema": "|".join(schema_cols)}

        out.update(keys_first)

        for k, v in row.items():
            if k == "column_schema" or k in keys_first:
                continue
            out[k] = v

        return out

    def optimize_sample(self, sample: RecordSample, TRI):
        """ Loop through each layer than neuron than weight"""
        is_batch_end = self.is_end_of_batch(sample, TRI)
        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id in range(len(neuron.weights)): self.optimize_weight(neuron, weight_id, sample, TRI)
            if is_batch_end:  neuron.accumulated_leverage = [0.0] * len(neuron.weights)

    def optimize_weight(self, neuron, weight_id: int, sample: RecordSample, TRI):
        "1)add standard fields to dict. 2)write to db 3)Apply adjustment to weight.  WE will follow the dumbasses and SUBTRACT The adjustment (unfortunately)"

        row = self.optimizer_brain(neuron, weight_id, TRI)

        row = self.inject_keys_first(
            row,
            run_id=TRI.run_id,
            epoch=sample.epoch,
            sample_id=sample.sample_id,
            nid=neuron.nid,
            weight_id=weight_id,
        )
        print(f"in optimize_weight: {row}")

        is_batch_end = self.is_end_of_batch(sample, TRI)

        if not is_batch_end:
            row["Adj"] = 0.0  # Log but don't apply

        self.weight_update_buffer.append(row)
        neuron.weights[weight_id] -= row["Adj"]

        if len(self.weight_update_buffer) >= self.buffer_limit:
            self.flush(TRI)


    def write_weight_update(self, record: dict):
        """
        Buffers weight update. Flushes when buffer reaches limit.
        Caller MUST call flush_weight_updates() at end of epoch/run.
        """
        if not self.TRI.should_record(RecordLevel.FULL): return

        #Add record to buffer
        self.weight_update_buffer.append(record)

        # Auto-flush if buffer full
        if len(self.weight_update_buffer) >= self.buffer_limit:  self.flush_weight_updates()

    # StrategyOptimizer

    def flush(self, TRI):
        """Write buffered weight updates to DB"""
        if not self.weight_update_buffer:
            return

        #if not TRI.should_record(RecordLevel.FULL):
        self.weight_update_buffer.clear()
        #    return

        sample_row = self.weight_update_buffer[0]
        columns = list(sample_row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        columns_str = ", ".join(columns)

        sql = f"INSERT INTO WeightAdjustments ({columns_str}) VALUES ({placeholders})"

        rows = [tuple(row[col] for col in columns) for row in self.weight_update_buffer]
        TRI.db.executemany(sql, rows, "weight adjustments")

        self.weight_update_buffer.clear()

    def flush_weight_updatesDELETEME(self):
        """Write all buffered records to database"""
        if not self.weight_update_buffer: return

        # Use first record to determine schema
        first_record = self.weight_update_buffer[0]
        keys = list(first_record.keys())
        fields = ", ".join(keys)

        # Build placeholders
        id_fields = {'run_id', 'epoch', 'sample_id', 'neuron_id', 'weight_id', 'batch_id'}
        placeholders = [
            "?" if key in id_fields else "CAST(? AS REAL)"
            for key in keys
        ]

        table_name = "WeightUpdates"
        sql = f"INSERT INTO {table_name} ({fields}) VALUES ({', '.join(placeholders)})"

        # Convert all records to rows (same key order)
        rows = [[self.convert_numpy(record[key]) for key in keys]
                for record in self.weight_update_buffer]

        self.db.executemany(sql, rows, "weight updates")
        self.weight_update_buffer.clear()

# ==============================================================================
# OPTIMIZER IMPLEMENTATIONS
# ==============================================================================

#def sgd_brain(neuron, weight_id, sample: RecordSample, TRI: TrainingRunInfo ):

def sgd_brain(neuron, weight_id, TRI):
    """Brain pulls what it needs, returns complete row as dict"""
    input_value     = neuron.inputs[weight_id]
    blame           = neuron.accepted_blame
    lr              = neuron.learning_rates[weight_id]
    before          = neuron.weights[weight_id]

    leverage        = input_value * blame
    adjustment      = lr * leverage
    after = before - adjustment

    return {
        "Input": input_value,
        "Blame": blame,
        "Leverage": leverage,
        "LR": lr,
        "Adj": adjustment,
        "Before": before,
        "After": after,
    }

Optimizer_SGD = StrategyOptimizer(
    name        = "Stochastic Gradient Descent",
    desc        = "Updates weights using the raw gradient scaled by learning rate.",
    when_to_use = "Simple problems, shallow networks, or when implementing your own optimizer.",
    best_for    = "Manual tuning, simple models, or teaching tools.",
    optimizer_brain = sgd_brain,
   #Optional for Adam -> state_per_weight=["m", "v"],6
    #Optional for Adam -> state_per_neuron=["t"],
)
