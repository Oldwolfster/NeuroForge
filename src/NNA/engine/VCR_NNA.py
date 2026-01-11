from src.NNA.utils.enums import RecordLevel


class VCR_NNA:
    def __init__(self, TRI):
        self.TRI = TRI
        self.weight_update_buffer = []
        self.buffer_limit = 5000

    def record_weight_update(self, record: dict):
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