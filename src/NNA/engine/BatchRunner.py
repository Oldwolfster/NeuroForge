from pathlib import Path

class BatchRunner:
    def __init__(self, batch_id: int, db_dsk):
        print("BR Constructor")
        self.batch_id = batch_id
        self.conn = db_dsk.conn
        self.pending_runs = []
        self.current_index = -1
        self.current_run_id = None
        self.load_pending_runs()


    def load_pending_runs(self):
        """Load all pending runs for this batch, ordered by run_id"""

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT run_id, seed, gladiator, arena, architecture,
                   loss, optimizer, hidden_activation, output_activation,
                   initializer, learning_rate, lr_specified
            FROM training_runs 
            WHERE batch_id = ? AND status = 'pending'
            ORDER BY run_id
        ''', (self.batch_id,))

        self.pending_runs = cursor.fetchall()
        print(f"🔍 BatchRunner: Found {len(self.pending_runs)} pending runs for batch_id={self.batch_id}")

    def __iter__(self):
        """Make BatchRunner iterable"""
        return self

    def __next__(self):
        """Return next setup dict, or raise StopIteration"""
        self.current_index += 1

        if self.current_index >= len(self.pending_runs):
            raise StopIteration

        row = self.pending_runs[self.current_index]
        self.current_run_id = row[0]  # run_id is first column

        # Build setup dict from DB row (strings for now, we'll deserialize next chunk)
        setup = {
            'seed': row[1],
            'gladiator': row[2],
            'arena': row[3],
            'architecture': eval(row[4]),  # "[4,4,1]" -> [4,4,1]
            'loss': row[5],  # Still string var_name
            'optimizer': row[6],
            'hidden_activation': row[7],
            'output_activation': row[8],
            'initializer': row[9],
            'learning_rate': row[10],
            'lr_specified': bool(row[11]),
        }

        return setup

    @property
    def current_run(self):
        """1-based current run number"""
        return self.current_index + 1

    @property
    def total_runs(self):
        """Total number of runs in this batch"""
        return len(self.pending_runs)