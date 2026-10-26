# ══════════════════════════════════════════════════════════════════════════════
# db_recorder.py    NEW FILE — buffered writes.  One path for every table.
# ══════════════════════════════════════════════════════════════════════════════
import json


class db_recorder:
    """Buffers rows during the epoch, writes them in one short transaction.

       The lock is held for the flush only — never across training compute.
    """

    ARG_COUNT   = 20                                    # WeightAdjustments arg_1..arg_20
    ROW_LIMIT   = 20000                                 # safety valve between epochs

    COLUMNS = {
        "RecordEpoch"       : ("run_id", "epoch", "phase", "correct", "wrong",
                               "accuracy", "mae"),

        "RecordSample"      : ("run_id", "epoch", "phase", "sample_id",
                               "inputs", "inputs_unscaled", "is_true",
                               "target", "target_unscaled",
                               "prediction", "prediction_unscaled", "prediction_raw",
                               "prediction_label", "loss_function", "loss", "loss_gradient"),

        "Neuron"            : ("run_id", "epoch", "sample_id", "nid", "layer_id",
                               "position", "num_inputs", "raw_sum", "activation_value",
                               "activation_gradient", "accepted_blame",
                               "neuron_inputs", "learning_rates", "accumulated_leverage"),

        "Weight"            : ("run_id", "epoch", "sample", "nid", "weight_id",
                               "value_before", "value"),

        # order mirrors the producer's tuple, not the DDL
        "ErrorSignalCalcs"  : ("epoch", "sample", "run_id", "nid", "weight_id",
                               "arg_1", "op_1", "arg_2", "op_2", "arg_3", "op_3", "result"),

        "WeightAdjustments" : ("run_id", "epoch", "sample_id", "nid", "weight_id")
                            + tuple(f"arg_{i}" for i in range(1, ARG_COUNT + 1)),
    }

    def __init__(self, db):
        self.db         = db
        self.buffers    = {table: [] for table in self.COLUMNS}
        self.sql        = {table: self.build_sql(table) for table in self.COLUMNS}
        self.row_count  = 0

    # db_recorder.py  method: build_sql  NEW:  one prepared insert per table
    def build_sql(self, table):
        columns      = self.COLUMNS[table]
        placeholders = ", ".join("?" * len(columns))
        return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

    # db_recorder.py  method: buffer_object  NEW:  pull named columns off an object
    def buffer_object(self, table, obj, **context):
        values = [context[col] if col in context else getattr(obj, col)
                  for col in self.COLUMNS[table]]
        self.buffer_row(table, values)

    # db_recorder.py  method: buffer_row  NEW:  already-ordered values
    def buffer_row(self, table, values):
        self.buffers[table].append(self.clean(values))
        self.row_count += 1
        if self.row_count >= self.ROW_LIMIT: self.flush()

    # db_recorder.py  method: clean  NEW:  numpy scalars out, json in
    def clean(self, values):
        return tuple(self.clean_value(v) for v in values)

    # db_recorder.py  method: clean_value  NEW:  one value
    def clean_value(self, value):
        if isinstance(value, (list, dict)):     return json.dumps(value)    # REFERENCE
        return value

    # db_recorder.py  method: flush  NEW:  every buffer, one short transaction
    def flush(self):
        if not self.row_count: return
        self.db.begin()
        try:
            for table, rows in self.buffers.items():
                if rows: self.db.executemany(self.sql[table], rows)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        finally:
            self.clear()

    # db_recorder.py  method: clear  NEW:  empty every buffer
    def clear(self):
        for rows in self.buffers.values(): rows.clear()
        self.row_count = 0

