import json
from typing import List
from .Neuron import Neuron
from src.NNA.utils.RamDB import RamDB
from .RecordEpoch import RecordEpoch
from .RecordSample import RecordSample
from .TrainingRunInfo import TrainingRunInfo, RecordLevel

#from src.NNA.engine.convergence.ConvergenceDetector import ConvergenceDetector


class VCR:
    def __init__(self, TRI):
        # Run Level members
        self.TRI : TrainingRunInfo  = TRI
        self.db  : RamDB            = TRI.db
        self.neurons                = Neuron.neurons
        self.batch_id               = 0
        self.sample          = 0                         # Current sample #
        self.epoch_curr_number      = 1                         # Which epoch are we currently on.
        self.sample_count           = len(TRI.training_data.get_list())          # Calculate and store sample count= 0               # Number of samples in each sample.
        #TODO self.converge_detector      = ConvergenceDetector(TRI.training_data, TRI.config)
        self.abs_error_for_epoch    = 0
        self.bd_correct             = 0
        self.convergence_signal     = None      # Will be set by convergence detector
        self.backpass_finalize_info = []

    def record_sample(self, record_sample: RecordSample, layers: List[List[Neuron]]):
        """Record sample data — optimizer handles its own weight update logging now"""
        self.abs_error_for_epoch += abs(record_sample.error_unscaled)
        if record_sample.is_true is True:
            self.bd_correct += 1

        if not self.TRI.should_record(RecordLevel.FULL):
            return

        self.TRI.db.add(record_sample)

        for layer_index, layer in enumerate(layers):
            for neuron in layer:
                if layer_index == 0:
                    raw_inputs = json.loads(record_sample.inputs)
                    neuron.neuron_inputs = [1.0] + raw_inputs
                else:
                    previous_layer = layers[layer_index - 1]
                    neuron.neuron_inputs = [1.0] + [prev.activation_value for prev in previous_layer]

                self.TRI.db.add(
                    neuron,
                    exclude_keys={"activation", "learning_rate", "weights", "weights_before"},
                    run_id=self.TRI.run_id,
                    epoch=record_sample.epoch,
                    sample_id=record_sample.sample_id
                )

        self.bulk_insert_weights(
            run_id=self.TRI.run_id,
            epoch=record_sample.epoch,
            sample=record_sample.sample_id
        )

    def finish_epoch(self, epoch: int):
        """End of epoch — flush optimizer buffer, record stats"""
        # Flush any remaining optimizer data
        self.TRI.config.optimizer.flush(self.TRI)

        self.TRI.last_epoch = epoch
        mae = self.abs_error_for_epoch / self.TRI.training_data.sample_count
        self.TRI.mae = mae

        if self.TRI.lowest_mae > mae:
            self.TRI.lowest_mae = mae
            self.TRI.lowest_mae_epoch = epoch

        self.TRI.bd_correct = self.bd_correct
        current_accuracy = self.TRI.accuracy

        if current_accuracy > self.TRI.best_accuracy:
            self.TRI.best_accuracy = current_accuracy
            self.TRI.best_accuracy_epoch = epoch

        epoch_record = RecordEpoch(
            run_id=self.TRI.run_id,
            epoch=epoch,
            correct=self.bd_correct,
            wrong=self.sample_count - self.bd_correct,
            accuracy=current_accuracy,
            mae=mae
        )
        self.TRI.db.add(epoch_record)

        self.abs_error_for_epoch = 0
        self.bd_correct = 0
        self.epoch_curr_number += 1

        return "Did Not Converge"  # TODO: convergence detector

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
            self.db.execute(sql_query, "Weight")


############# Record Backpass info for pop up window of NeuroForge #############
    def record_weight_updates(self, weight_update_metrics, update_or_finalize: str):
        """
        Inserts weight update calculations for the current sample into the database.
        Compatible with arbitrary arg/op chains.
        """
        if not self.TRI.should_record(RecordLevel.FULL ): return
        if not weight_update_metrics: return
        sample_row = weight_update_metrics[0]
        fields = self.build_weight_update_field_list(sample_row)
        placeholders = self.build_weight_update_placeholders(sample_row)

        table_name = f"WeightAdjustments_{update_or_finalize}_{self.TRI.run_id}" #TODO susceptible to SQL injection
        sql = f"""
            INSERT INTO {table_name}
            ({fields})
            VALUES ({placeholders})
        """

        converted_rows = [self.convert_numpy_scalars_because_python_is_shit(row) for row in weight_update_metrics]

        #print(f"Data about to be  INSERT{converted_rows}")
        #print("Below is table content")
        #self.TRI.db.query_print(f"Select * from {table_name}")

        self.TRI.db.executemany(sql, converted_rows,"weight adjustments")
        #print("Insert worked")
        weight_update_metrics.clear()

    def convert_numpy_scalars_because_python_is_shit(self, row):
        """
        Converts any NumPy scalar values in the given row to their native Python types.
        Friggen ridiculous it was converting either 0 to null or 1 to 0.... what a joke this language is
        """
        return [x.item() if hasattr(x, 'item') else x for x in row]

    def build_weight_update_field_list(self, sample_row):
        #base_fields = ["epoch", "sample", "model_id", "nid", "weight_index", "batch_id"]
        base_fields = ["epoch", "sample", "nid", "weight_index", "batch_id"]
        custom_fields = []
        # Now create one custom field per element after the first six (base fields).
        for i in range(5, len(sample_row)):
            arg_n = i - 5 + 1
            custom_fields.append(f"arg_{arg_n}")
        return ", ".join(base_fields + custom_fields)

    def build_weight_update_placeholders(self, sample_row):
        #base_placeholders = ["?"] * 6
        base_placeholders = ["?"] * 5
        arg_op_placeholders = []

        #for i in range(6, len(sample_row)):
        for i in range(5, len(sample_row)):
            arg_op_placeholders.append("CAST(? AS REAL)")  # arg
        return ", ".join(base_placeholders + arg_op_placeholders)

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

    def bulk_insert_weights(self,run_id, epoch, sample):
        """
        Collects all weight values across neurons and creates a bulk insert SQL statement.
        """
        sql_statements = []
        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id, (prev_weight, weight) in enumerate(zip(neuron.weights_before, neuron.weights)):
                    sql_statements.append(
                        f"({run_id}, {epoch}, {sample}, {neuron.nid}, {weight_id}, {prev_weight}, {weight})"
                    )

        if sql_statements:
            sql_query = f"INSERT INTO Weight (run_id, epoch, sample, nid, weight_id, value_before, value) VALUES {', '.join(sql_statements)};"
            self.db.execute(sql_query, "Weight")