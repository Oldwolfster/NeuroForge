from pathlib import Path

from src.NNA.legos._LegoManager import LegoManager


class BatchRunner:

    def __iter__(self):
        """Make BatchRunner iterable"""
        return self

    def __init__(self, batch_id: int, hyper):
        #print("BR Constructor")
        self.lego_mgr = LegoManager()
        self.batch_id = batch_id
        self.conn = hyper.db_dsk.conn
        self.current_index = -1
        self.current_run_id = None
        if hyper.neuro_FORGE[0] == 0:   self.load_pending_run_ids()
        else:                           self.pending_run_ids = hyper.neuro_FORGE

    def load_pending_run_ids(self):
        """Load only the run_ids, not the full configs"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT run_id 
            FROM batch_history
            WHERE batch_id = ? AND status = 'pending'
            ORDER BY run_id
        ''', (self.batch_id,))

        self.pending_run_ids = [row[0] for row in cursor.fetchall()]
        print(f"🔍 BatchRunner: Found {len(self.pending_run_ids)} pending runs for batch_id={self.batch_id}")

    def __next__(self):
        """Fetch next run on-demand and return (run_id, setup)"""
        self.current_index += 1

        if self.current_index >= len(self.pending_run_ids):
            raise StopIteration

        self.current_run_id = self.pending_run_ids[self.current_index]
        print(f"self.current_run_id={self.current_run_id}")
        # Fetch config for this run_id only
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT key, value
            FROM batch_details
            WHERE run_id = ?
        ''', (self.current_run_id,))


        run_config = dict(cursor.fetchall())
        setup = self.deserialize_run_config(run_config)

        return (self.current_run_id, setup)

    def deserialize_run_config(self, run_config: dict) -> dict:
        """Convert string values back to proper types"""
        setup = {}

        for key, value in run_config.items():
            if key == 'architecture':
                setup[key] = eval(value)  # "[4,4,1]" -> [4,4,1]
            elif key in ['seed', 'batch_size']:
                setup[key] = int(value)
            elif key == 'learning_rate':
                setup[key] = float(value) if value != 'None' else None
            elif key == 'lr_specified':
                setup[key] = value == 'True'
            elif key in ['gladiator', 'arena']:
                setup[key] = value
            elif self.lego_mgr.is_lego_dimension(key) and value and value != 'None':
                setup[key] = self.lego_mgr.string_to_lego(value)
            else:
                setup[key] = None

        return setup

    @property
    def current_run(self):
        """1-based current run number"""
        return self.current_index + 1

    @property
    def total_runs(self):
        """Total number of runs in this batch"""
        return len(self.pending_run_ids)