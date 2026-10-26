# ══════════════════════════════════════════════════════════════════════════════
# db_schema.py      NEW FILE — sole owner of the .nf schema
# ══════════════════════════════════════════════════════════════════════════════
from src.NNA.utils.db_conn import db_conn
from src.NNA.utils.db_catalog import db_catalog

class db_schema:
    """Every table in a .nf file is declared here and nowhere else.

       create_database() is the only entry point.  It is idempotent and safe to
       call from every process on every start: drop a table for an alter and the
       next run rebuilds exactly that one.
    """

    # db_schema.py  method: create_database  NEW:  sole entry point — returns a ready db_conn
    @staticmethod
    def create_database(db_path):
        db = db_conn(db_path)
        db_schema.create_tables(db)
        db_schema.create_triggers(db)
        db_catalog.populate_all(db)
        return db


    # db_schema.py  method: create_triggers  NEW:  stamp completed_at when status flips
    @staticmethod
    def create_triggers(db):
        db.execute("""
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
            END""")

    @staticmethod
    def create_tables(db):

        # ── WITHOUT ROWID — small fixed-width rows, well under the 4K threshold ──

        db.execute("""
            CREATE TABLE IF NOT EXISTS RecordEpoch (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                phase                   INTEGER NOT NULL,
                correct                 INTEGER,
                wrong                   INTEGER,
                accuracy                REAL,
                mae                     REAL,
                PRIMARY KEY (run_id, epoch, phase)
            ) WITHOUT ROWID""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS Weight (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample                  INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                weight_id               INTEGER NOT NULL,
                value_before            REAL    NOT NULL,
                value                   REAL    NOT NULL,
                PRIMARY KEY (run_id, epoch, sample, nid, weight_id)
            ) WITHOUT ROWID""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS ErrorSignalCalcs (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample                  INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                weight_id               INTEGER NOT NULL,
                arg_1                   REAL    NOT NULL,
                op_1                    TEXT    NOT NULL CHECK (op_1 IN ('+', '-', '*', '/', '=')),
                arg_2                   REAL    NOT NULL,
                op_2                    TEXT    NOT NULL CHECK (op_2 IN ('+', '-', '*', '/', '=')),
                arg_3                   REAL    DEFAULT NULL,
                op_3                    TEXT    DEFAULT NULL CHECK (op_3 IN ('+', '-', '*', '/', '=')),
                result                  REAL    NOT NULL,
                PRIMARY KEY (run_id, epoch, sample, nid, weight_id)
            ) WITHOUT ROWID""")

        # ── rowid tables + PK — json/wide rows.  PK still builds the B-tree index. ──

        db.execute("""
            CREATE TABLE IF NOT EXISTS Neuron (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample_id               INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                layer_id                INTEGER,
                position                INTEGER,
                num_inputs              INTEGER,
                raw_sum                 REAL,
                activation_value        REAL,
                activation_gradient     REAL,
                accepted_blame          REAL,
                neuron_inputs           TEXT,
                learning_rates          TEXT,
                accumulated_leverage    TEXT,
                PRIMARY KEY (run_id, epoch, sample_id, nid)
            )""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS RecordSample (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                phase                   INTEGER NOT NULL,
                sample_id               INTEGER NOT NULL,
                inputs                  TEXT,
                inputs_unscaled         TEXT,
                is_true                 INTEGER,
                target                  REAL,
                target_unscaled         REAL,
                prediction              REAL,
                prediction_unscaled     REAL,
                prediction_raw          REAL,
                prediction_label        TEXT,
                loss_function           TEXT,
                loss                    REAL,
                loss_gradient           REAL,

                error                     REAL    GENERATED ALWAYS AS (target - prediction_raw) VIRTUAL,
                error_unscaled            REAL    GENERATED ALWAYS AS (target_unscaled - prediction_unscaled) VIRTUAL,
                absolute_error            REAL    GENERATED ALWAYS AS (abs(target - prediction_raw)) VIRTUAL,
                absolute_error_unscaled   REAL    GENERATED ALWAYS AS (abs(target_unscaled - prediction_unscaled)) VIRTUAL,
                squared_error             REAL    GENERATED ALWAYS AS ((target - prediction_raw) * (target - prediction_raw)) VIRTUAL,
                relative_error            REAL    GENERATED ALWAYS AS (abs((target - prediction_raw) / (target + 1e-64))) VIRTUAL,
                is_false                  INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 0                   THEN 1 ELSE 0 END) VIRTUAL,
                is_true_positive          INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 1 AND target <> 0    THEN 1 ELSE 0 END) VIRTUAL,
                is_true_negative          INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 1 AND target  = 0    THEN 1 ELSE 0 END) VIRTUAL,
                is_false_positive         INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 0 AND target  = 0    THEN 1 ELSE 0 END) VIRTUAL,
                is_false_negative         INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 0 AND target <> 0    THEN 1 ELSE 0 END) VIRTUAL,

                PRIMARY KEY (run_id, epoch, phase, sample_id)
            )""")

        arg_columns = ",\n                ".join(
            f"arg_{i:<2}                  REAL" for i in range(1, 21))
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS WeightAdjustments (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample_id               INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                weight_id               INTEGER NOT NULL,
                {arg_columns},
                PRIMARY KEY (run_id, epoch, sample_id, nid, weight_id)
            )""")




    @staticmethod
    def create_tables(db):

        db.execute("""
            CREATE TABLE IF NOT EXISTS batch (
                batch_id                INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name              TEXT,
                batch_notes             TEXT,
                config                  TEXT,
                expected_runs           INTEGER,
                created_at              DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS batch_history (
                run_id                  INTEGER PRIMARY KEY,
                seed                    INTEGER,
                status                  TEXT DEFAULT 'pending',
                gladiator               TEXT,
                arena                   TEXT,
                epoch_count             INTEGER,
                checkpoint_phase        INTEGER,                
                checkpoint_epoch        INTEGER,
                
                

                mae                     REAL,          --  test     
                accuracy                REAL,          --  test     
                sample_count_test       INTEGER,                    
                mae_train               REAL,                       
                mae_epoch_train         INTEGER,                    
                sample_count_train      INTEGER,                    
                mae_validate            REAL,                       
                mae_epoch_validate      INTEGER,                    
                sample_count_validate   INTEGER, 
                
                convergence_condition   TEXT,
                runtime_seconds         REAL,
                learning_rate           REAL,
                lr_specified            INTEGER,
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
                record_level            INTEGER,       
                input_count             INTEGER,       --
                feature_labels          TEXT,          --  json list
                source_code             TEXT,          --  arena source
                backprop_headers        TEXT,          --  json list
                scaler_params           TEXT,          --  json list of dicts (fitted)      
                error_text              TEXT,          --  exception type + message          
                batch_id                INTEGER,
                created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at            DATETIME
            )""")

        # ── WITHOUT ROWID — small fixed-width rows, well under the 4K threshold ──

        db.execute("""
            CREATE TABLE IF NOT EXISTS RecordEpoch (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                phase                   INTEGER NOT NULL,
                correct                 INTEGER,
                wrong                   INTEGER,
                accuracy                REAL,
                mae                     REAL,
                PRIMARY KEY (run_id, epoch, phase)
            ) WITHOUT ROWID""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS Weight (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample                  INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                weight_id               INTEGER NOT NULL,
                value_before            REAL    NOT NULL,
                value                   REAL    NOT NULL,
                PRIMARY KEY (run_id, epoch, sample, nid, weight_id)
            ) WITHOUT ROWID""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS ErrorSignalCalcs (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample                  INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                weight_id               INTEGER NOT NULL,
                arg_1                   REAL    NOT NULL,
                op_1                    TEXT    NOT NULL CHECK (op_1 IN ('+', '-', '*', '/', '=')),
                arg_2                   REAL    NOT NULL,
                op_2                    TEXT    NOT NULL CHECK (op_2 IN ('+', '-', '*', '/', '=')),
                arg_3                   REAL    DEFAULT NULL,
                op_3                    TEXT    DEFAULT NULL CHECK (op_3 IN ('+', '-', '*', '/', '=')),
                result                  REAL    NOT NULL,
                PRIMARY KEY (run_id, epoch, sample, nid, weight_id)
            ) WITHOUT ROWID""")

        # ── rowid tables + PK — json/wide rows.  PK still builds the B-tree index. ──

        db.execute("""
            CREATE TABLE IF NOT EXISTS RecordSample (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                phase                   INTEGER NOT NULL,
                sample_id               INTEGER NOT NULL,
                inputs                  TEXT,
                inputs_unscaled         TEXT,
                is_true                 INTEGER,
                target                  REAL,
                target_unscaled         REAL,
                prediction              REAL,
                prediction_unscaled     REAL,
                prediction_raw          REAL,
                prediction_label        TEXT,
                loss_function           TEXT,
                loss                    REAL,
                loss_gradient           REAL,

                error                     REAL    GENERATED ALWAYS AS (target - prediction_raw) VIRTUAL,
                error_unscaled            REAL    GENERATED ALWAYS AS (target_unscaled - prediction_unscaled) VIRTUAL,
                absolute_error            REAL    GENERATED ALWAYS AS (abs(target - prediction_raw)) VIRTUAL,
                absolute_error_unscaled   REAL    GENERATED ALWAYS AS (abs(target_unscaled - prediction_unscaled)) VIRTUAL,
                squared_error             REAL    GENERATED ALWAYS AS ((target - prediction_raw) * (target - prediction_raw)) VIRTUAL,
                relative_error            REAL    GENERATED ALWAYS AS (abs((target - prediction_raw) / (target + 1e-64))) VIRTUAL,
                is_false                  INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 0                   THEN 1 ELSE 0 END) VIRTUAL,
                is_true_positive          INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 1 AND target <> 0    THEN 1 ELSE 0 END) VIRTUAL,
                is_true_negative          INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 1 AND target  = 0    THEN 1 ELSE 0 END) VIRTUAL,
                is_false_positive         INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 0 AND target  = 0    THEN 1 ELSE 0 END) VIRTUAL,
                is_false_negative         INTEGER GENERATED ALWAYS AS (CASE WHEN is_true = 0 AND target <> 0    THEN 1 ELSE 0 END) VIRTUAL,

                PRIMARY KEY (run_id, epoch, phase, sample_id)
            )""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS Neuron (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample_id               INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                layer_id                INTEGER,
                position                INTEGER,
                num_inputs              INTEGER,
                raw_sum                 REAL,
                activation_value        REAL,
                activation_gradient     REAL,
                accepted_blame          REAL,
                neuron_inputs           TEXT,
                learning_rates          TEXT,
                accumulated_leverage    TEXT,
                PRIMARY KEY (run_id, epoch, sample_id, nid)
            )""")

        arg_columns = ",\n                ".join(
            f"arg_{i:<2}                  REAL" for i in range(1, 21))
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS WeightAdjustments (
                run_id                  INTEGER NOT NULL,
                epoch                   INTEGER NOT NULL,
                sample_id               INTEGER NOT NULL,
                nid                     INTEGER NOT NULL,
                weight_id               INTEGER NOT NULL,
                {arg_columns},
                PRIMARY KEY (run_id, epoch, sample_id, nid, weight_id)
            )""")

        # ── unchanged ──

        db.execute("""
            CREATE TABLE IF NOT EXISTS lego (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                name                    TEXT,
                dimensions              TEXT,
                desc                    TEXT,
                when_to_use             TEXT,
                best_for                TEXT,
                code                    TEXT
            )""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS log_lr_sweep (
                batch_id                INTEGER,
                run_id                  INTEGER,
                seq                     INTEGER,
                learning_rate           REAL,
                mae                     REAL,
                phase                   TEXT,
                is_best                 INTEGER DEFAULT 0,
                message                 TEXT,
                PRIMARY KEY (batch_id, run_id, seq)
            )""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS log_automl (
                batch_id                INTEGER,
                run_id                  INTEGER,
                seq                     INTEGER,
                priority                TEXT,
                setting                 TEXT,
                value                   TEXT,
                applied                 INTEGER,
                override                INTEGER,
                condition               TEXT,
                message                 TEXT,
                PRIMARY KEY (batch_id, run_id, seq)
            )""")

        db.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                obs_id                  INTEGER PRIMARY KEY,
                target_type             TEXT    NOT NULL,
                target_id               INTEGER NOT NULL,
                note_text               TEXT    NOT NULL,
                created_at              DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")