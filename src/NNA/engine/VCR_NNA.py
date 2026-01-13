import json

from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.RecordEpoch import RecordEpoch
from src.NNA.engine.RecordSample import RecordSample
from typing import TYPE_CHECKING
if TYPE_CHECKING:     from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from src.NNA.utils.enums import RecordLevel

class VCR_NNA:
    def __init__(self, TRI):
        self.TRI : TrainingRunInfo  = TRI
        self.weight_update_buffer   = []
        self.buffer_limit           = 5000

    def write_epoch(self, record_sample):
        self.flush()
        self.TRI.db.add(record_sample)


    def write_sample(self, record_sample: RecordSample, blame_calculations):

        #1) Record sample
        self.TRI.db.add(record_sample)

        #2) Record Blame calculations (bottom right quadrant of popup)
        self.record_blame_calculations(blame_calculations)

        #3) Record Neurons
        for layer_index, layer in enumerate(Neuron.layers):
            for neuron in layer:
                self.TRI.db.add(
                    neuron,
                    exclude_keys={"optimizer_state","activation", "learning_rate", "weights", "weights_before"},
                    run_id=self.TRI.run_id,
                    epoch=record_sample.epoch,
                    sample_id=record_sample.sample_id
                )

        #4) Record Weights
        self.bulk_insert_weights(
            run_id=self.TRI.run_id,
            epoch=record_sample.epoch,
            sample=record_sample.sample_id
        )


    def bulk_insert_weights(self, run_id, epoch, sample):
        """Bulk insert all weight values"""
        sql_statements = []
        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id, (prev_weight, weight) in enumerate(zip(neuron.weights_before, neuron.weights)):
                    sql_statements.append(
                        f"({run_id}, {epoch}, {sample}, {neuron.nid}, {weight_id}, {prev_weight}, {weight})"
                    )

        if sql_statements:
            sql_query = f"INSERT INTO Weight (run_id, epoch, sample, nid, weight_id, value_before, value) VALUES {', '.join(sql_statements)};"
            self.TRI.db.execute(sql_query, "Weight")


    def record_optimizer_logic(self, record: dict):
        """Buffer one weight update record"""
        if not self.TRI.should_record(RecordLevel.FULL): return

        # Capture headers on first record
        if self.TRI.backprop_headers is None:
            excluded = ['run_id', 'epoch', 'sample_id', 'nid', 'weight_id']
            self.TRI.backprop_headers = [k for k in record.keys() if k not in excluded]

        self.weight_update_buffer.append(record)
        if len(self.weight_update_buffer) >= self.buffer_limit:
            self.flush()

    def flush(self):
        """Write buffer to DB"""
        if not self.weight_update_buffer: return

        sample_row = self.weight_update_buffer[0]
        columns = list(sample_row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        columns_str = ", ".join(columns)

        sql = f"INSERT INTO WeightAdjustments ({columns_str}) VALUES ({placeholders})"
        rows = [tuple(row[col] for col in columns) for row in self.weight_update_buffer]

        self.TRI.db.executemany(sql, rows, "weight adjustments")
        self.weight_update_buffer.clear()

    def convert_numpy_scalars_because_python_is_shit(self, row):
        """
        Converts any NumPy scalar values in the given row to their native Python types.
        Friggen ridiculous it was converting either 0 to null or 1 to 0.... what a joke this language is
        """
        return [x.item() if hasattr(x, 'item') else x for x in row]


    def record_blame_calculations(self, blame_calculations):
        """
        Inserts all backprop calculations for the current sample into the database.
        """

        #print("********  Distribute Error Calcs************")
        #for row in self.blame_calculations:
        #    print(row)
        if not self.TRI.should_record(RecordLevel.FULL ): return
        sql = """
        INSERT INTO ErrorSignalCalcs
        (epoch, sample, run_id, nid, weight_id, 
         arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
        VALUES 
        (?, ?, ?, ?, ?, 
         CAST(? AS REAL), ?, 
         CAST(? AS REAL), ?, 
         CAST(? AS REAL), ?, 
         CAST(? AS REAL))
        """

        # Convert each row to ensure any numpy scalars are native Python types
        converted_rows = [self.convert_numpy_scalars_because_python_is_shit(row) for row in blame_calculations]
        #print(f"BLAME {self.blame_calculations}")

        #Heads up, sometimes overflow error look like key violation here

        self.TRI.db.executemany(sql, blame_calculations, "error signal")
        blame_calculations.clear()
