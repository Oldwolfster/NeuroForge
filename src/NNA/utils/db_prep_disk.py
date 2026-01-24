'''
                problem_type TEXT,
                seed INTEGER,
                sample_count INTEGER,
                target_min REAL,
                target_max REAL,
                target_min_label TEXT,
                target_max_label TEXT,
                target_mean REAL,
                target_stdev REAL,
                notes TEXT,
                rerun_config TEXT
SEED AND GIT COMMIT

'''


def check_batch_schema(conn):
    cursor = conn.cursor()

    # Table 1: Batch specifications (what to sweep)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batch_specs (
            batch_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name      TEXT,
            batch_notes     TEXT,
            dimensions      TEXT,
            gladiators      TEXT,
            arenas          TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table 2: Expanded run configurations (key-value pairs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batch_details (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id    INTEGER,
            run_id      INTEGER,
            key         TEXT,
            value       TEXT
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_batch_details_lookup
        ON batch_details(batch_id, run_id)
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batch_history (
            run_id                  INTEGER PRIMARY KEY,
            seed                    INTEGER,            
            status                  TEXT DEFAULT 'pending',
            gladiator               TEXT,
            arena                   TEXT,
            epoch_count             INTEGER,
            accuracy                REAL,
            final_mae               REAL,
            best_mae                REAL,                        
            convergence_condition   TEXT,
            runtime_seconds         REAL,            
            learning_rate           REAL,
            batch_size              INTEGER,
            architecture            TEXT,
            optimizer               TEXT,
            weight_initializer      TEXT,
            loss_function           TEXT,
            hidden_activation       TEXT,
            output_activation       TEXT,
            target_scaler           TEXT,
            input_scalers           TEXT,
            problem_type            TEXT,
            sample_count            INTEGER,
            target_min              REAL,
            target_max              REAL,
            target_min_label        TEXT,
            target_max_label        TEXT,
            target_mean             REAL,
            target_stdev            REAL,
            batch_id                INTEGER,            
            created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at            DATETIME 
                         
        )
    ''')
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS trg_batch_history_completed_at
    AFTER UPDATE OF status ON batch_history
    FOR EACH ROW
    WHEN NEW.status = 'completed'
     AND (OLD.status IS NULL OR OLD.status <> 'completed')
     AND NEW.completed_at IS NULL
    BEGIN
      UPDATE batch_history
         SET completed_at = CURRENT_TIMESTAMP
       WHERE run_id = NEW.run_id;
    END;
    ''')

def add_pytorch_views(conn):
    cursor = conn.cursor()

    cursor.execute('''
    CREATE VIEW IF NOT EXISTS last AS
        SELECT *
        FROM batch_history
        WHERE batch_id = (SELECT MAX(batch_id) FROM batch_history)
        ORDER BY run_id DESC
    ''')

    cursor.execute('''
    CREATE VIEW IF NOT EXISTS v_nna AS
SELECT
  w.run_id,
  w.epoch,
  w.nid,
  w.weight_id,
  w.value
FROM weight w
JOIN (
  SELECT run_id, epoch, nid, weight_id, MAX(sample) AS sample
  FROM weight
  GROUP BY run_id, epoch, nid, weight_id
) last
  ON  last.run_id    = w.run_id
  AND last.epoch     = w.epoch
  AND last.nid       = w.nid
  AND last.weight_id = w.weight_id
  AND last.sample    = w.sample;

    ''')

    cursor.execute('''
    CREATE VIEW IF NOT EXISTS v_pytorch AS
SELECT
  epoch,
  nid,
  weight_index AS weight_id,
  weight_value AS value
FROM pytorch_weights;

    ''')
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS v_nna_epoch AS
SELECT
  run_id,
  epoch,
  nid,
  weight_id,
  value
FROM weight
WHERE sample = 1;

    ''')

    cursor.execute('''
CREATE VIEW IF NOT EXISTS v_pytorch_epoch AS
SELECT
  epoch,
  nid,
  weight_index AS weight_id,
  weight_value AS value
FROM pytorch_weights;

    ''')

    cursor.execute('DROP TABLE IF EXISTS pytorch_weights;')

    # Create table if not exists",
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pytorch_weights (
            epoch INTEGER,
            layer INTEGER,
            neuron INTEGER,
            nid INTEGER,
            weight_index INTEGER,
            weight_value REAL,
            PRIMARY KEY (epoch, layer, neuron, weight_index)
        )
    ''')
    conn.commit()
