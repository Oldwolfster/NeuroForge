from src.NNA.Legos.Activation import Activation_NoDamnFunction
from src.NNA.Legos.Initializer import Initializer_Tiny
from src.NNA.engine.Neuron import Neuron
from src.NNA.utils.RamDB import RamDB
from src.NNA.engine.RecordSample import RecordSample


def create_weight_adjustments_table(db: RamDB, run_id: int, update_or_finalize: str, arg_count=12):
    """
    Creates a dedicated WeightAdjustments_<run_id> table with arg_1..arg_N fields.
    """
    table_name = f"WeightAdjustments_{update_or_finalize}_{run_id}"
    fields = ",\n".join([f"    arg_{i + 1} REAL DEFAULT NULL" for i in range(arg_count)])

    sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch        INTEGER NOT NULL,
            sample   INTEGER NOT NULL,
            nid          INTEGER NOT NULL,
            -- model_id     TEXT NOT NULL, removed - model is part of table name... why have column with 1 unique value??
            weight_index INTEGER NOT NULL,
            batch_id     INTEGER NOT NULL DEFAULT 0,
            {fields}

        );
    """

    db.execute(sql)

    db.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_batch_lookup_{update_or_finalize}_{run_id}
        ON {table_name} (epoch, batch_id, nid, weight_index);
    """)


def prep_RamDB():
    db = RamDB()

    # Create dummy records to create table so we can create the view and indexes

    dummy_sample = RecordSample(is_true=0, run_id=0, epoch=0, sample_id=0, inputs="", target=0.1, prediction=0.1,
                                inputs_unscaled="", target_unscaled=0.1, prediction_unscaled=0.1, prediction_raw=0.1,
                                loss=0.1, loss_gradient=0.1, loss_function="dummy", accuracy_threshold=0.0, prediction_label="eatme")
    dummy_neuron = Neuron(0, 1, 0.0, Initializer_Tiny, 0,activation=Activation_NoDamnFunction)
    db.add(dummy_sample)

    db.add(dummy_neuron, exclude_keys={"activation", "output_neuron"}, run_id=0, epoch=0, sample_id=0)
    # db.execute("CREATE INDEX idx_model_epoch_sample ON Neuron (model, epoch, sample);")
    db.execute("CREATE INDEX idx_epoch_sample ON Neuron (run_id, epoch, sample_id);")
    db.execute("CREATE INDEX idx__RecordSample ON RecordSample (run_id,  sample_id);")



    # This feels not DRY --> epoch_create_view_epochSummary(db)
    db.execute("DELETE FROM RecordSample")  # Delete dummy records
    db.execute("DELETE FROM Neuron")  # Delete dummy records
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS Weight (
            run_id INT NOT NULL,
            epoch INTEGER NOT NULL,
            sample INTEGER NOT NULL,
            nid INTEGER NOT NULL,
            weight_id INTEGER NOT NULL,
            value_before REAL NOT NULL,
            value REAL NOT NULL,            
            PRIMARY KEY (run_id, epoch, sample, nid, weight_id)       
        );""")

    db.execute("""
        CREATE TABLE IF NOT EXISTS ErrorSignalCalcs (
            run_id       INTEGER NOT NULL,
            epoch        INTEGER NOT NULL,
            sample    INTEGER NOT NULL,            
            nid          INTEGER NOT NULL,
            weight_id    INTEGER NOT NULL,
            arg_1        REAL NOT NULL,
            op_1         TEXT NOT NULL CHECK (op_1 IN ('+', '-', '*', '/', '=')),  
            arg_2        REAL NOT NULL,
            op_2         TEXT NOT NULL CHECK (op_2 IN ('+', '-', '*', '/', '=')),
            arg_3        REAL DEFAULT NULL,
            op_3         TEXT DEFAULT NULL CHECK (op_3 IN ('+', '-', '*', '/', '=')),
            result       REAL NOT NULL,
            PRIMARY KEY (run_id, epoch, sample,  nid,weight_id)  -- Ensures unique calculations per neuron per step
        );""")

    return db


def create_weight_tables(db, run_id):
    create_weight_adjustments_table(db, run_id, "update")
    create_weight_adjustments_table(db, run_id, "finalize")
    delete_records(db, run_id)  # in case it had been run by LR sweep


def delete_records(db, run_id, possible_columns=None):
    """
    Deletes records across all tables where one of the possible columns matches the given gladiator.

    Args:
        db: Your database connection or wrapper.
        gladiator (str): The model ID or name to delete.
        possible_columns (list of str, optional): Columns to check, in order of preference.
    """
    if possible_columns is None:
        possible_columns = ['run_id', 'model', 'gladiator']

    # Delete tables that have run_id in name rather than waste a column
    table_name = f"WeightAdjustments_update_{run_id}"
    db.execute(f"DELETE FROM {table_name}")
    table_name = f"WeightAdjustments_finalize_{run_id}"
    db.execute(f"DELETE FROM {table_name}")

    # Get list of all table names
    tables = db.query("SELECT name FROM sqlite_master WHERE type='table';")

    for table_row in tables:
        # ez_debug(table_row=table_row)
        table_name = table_row['name']

        # Get column names for this table
        columns = db.query(f"PRAGMA table_info({table_name});", as_dict=False)
        column_names = [col[1] for col in columns]

        # Find first matching column
        matching_column = next((col for col in possible_columns if col in column_names), None)

        if matching_column:
            # print(f"🧹 Deleting from {table_name} where {matching_column} = '{gladiator}'")
            # db.execute(f"DELETE FROM {table_name} WHERE {matching_column} = ?", (gladiator,))
            db.execute(f"DELETE FROM {table_name} WHERE {matching_column} = '{run_id}'")

