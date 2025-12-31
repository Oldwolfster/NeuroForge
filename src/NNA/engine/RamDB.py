import json
import sqlite3
#import numpy as np
#from tabulate import tabulate
from datetime import datetime
from pathlib import Path
import csv
import os
import inspect
import time
class RamDB:
    # RamDB.py

    def __init__(self, path=None):
        self.tables = {}  # To track schemas for validation

        if path is None:
            db_path = ':memory:'
            self.is_memory = True
        else:
            path = Path(path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(path)
            self.is_memory = False

        self.conn = sqlite3.connect(db_path, isolation_level=None)
        self.cursor = self.conn.cursor()

        # Common PRAGMAs
        self.cursor.execute("PRAGMA page_size = 16384")
        self.cursor.execute("PRAGMA cache_size = -20000")
        self.cursor.execute("PRAGMA temp_store = MEMORY")
        #self.cursor.execute("PRAGMA locking_mode = EXCLUSIVE")
        self.cursor.execute("PRAGMA count_changes = OFF")

        if self.is_memory:
            # Aggressive settings - fine for RAM (no crash risk)
            self.cursor.execute("PRAGMA synchronous = OFF")
            self.cursor.execute("PRAGMA journal_mode = OFF")
            self.cursor.execute("PRAGMA cache_spill = OFF")
        else:
            # Safer for disk - won't corrupt on crash
            self.cursor.execute("PRAGMA synchronous = NORMAL")
            self.cursor.execute("PRAGMA journal_mode = WAL")


    def _infer_schema(self, obj, exclude_keys=None):
        """
        Infer SQLite schema from a Python object, including properties.
        """

        if exclude_keys is None:
            exclude_keys = set()
        else:
            exclude_keys = set(exclude_keys)

        schema = {}
        # Get all attributes, including properties
        for attr_name in dir(obj):
            # Skip private and special attributes
            if attr_name.startswith("_") or attr_name in exclude_keys:
                continue

            # Get the attribute or property value
            attr_value = getattr(obj, attr_name, None)

            if callable(attr_value):
                continue  # Skip methods

            # Handle various data types
            if isinstance(attr_value, bool):            schema[attr_name] = "INTEGER"  # Map bool to INTEGER
            elif isinstance(attr_value, int):           schema[attr_name] = "INTEGER"
            elif isinstance(attr_value, float):         schema[attr_name] = "REAL"
            elif isinstance(attr_value, str):           schema[attr_name] = "TEXT"
            elif isinstance(attr_value, (list, dict)):  schema[attr_name] = "TEXT"  # Serialize as JSON
            else:                                       raise TypeError(f"Unsupported field type: {type(attr_value)} for attribute '{attr_name}'")
        return schema

    def _create_table(self, table_name, schema): #I think this is being ignored due to another version lower
        """
        Create a SQLite table dynamically with context fields prioritized and a composite primary key.
        """
        # Define context fields and reorder schema
        context_fields = ['epoch', 'iteration']
        reordered_schema = {key: schema[key] for key in context_fields if key in schema}
        reordered_schema.update({key: schema[key] for key in schema if key not in context_fields})

        # Add composite primary key
        columns = ", ".join([f"{name} {type}" for name, type in reordered_schema.items()])
        primary_key = "PRIMARY KEY (epoch, iteration, nid)"
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}, {primary_key});"

        # Execute table creation
        #print(f"sql={sql}")
        self.cursor.execute(sql)
        self.tables[table_name] = schema  # Track schema


    def add(self, obj, exclude_keys=None, **context):
        """
        Add an object to the database with any number of context fields.
        Automatically creates the table if it does not exist.
        """
        # ─── start stopwatch ───────────────────────────────────────────────────────
        if not hasattr(self, '_add_timing'):
            self._add_timing = {}
        start = time.perf_counter()

        # ─── original add logic ────────────────────────────────────────────────────
        table_name = obj.__class__.__name__

        if table_name not in self.tables:
            self.create_table(table_name, obj, context, exclude_keys)

        if exclude_keys is None:
            data = {**context, **vars(obj)}
        else:
            data = {
                key: value for key, value in {**context, **vars(obj)}.items()
                if key not in exclude_keys
            }

        computed_fields = {
            attr: getattr(obj, attr)
            for attr in dir(obj)
            if isinstance(getattr(type(obj), attr, None), property)
        }
        data.update(computed_fields)

        for key, value in data.items():
            if isinstance(value, (list, dict)):
                data[key] = json.dumps(value)

        columns     = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        sql         = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"

        self.cursor.execute(sql, tuple(data.values()))
        # ────────────────────────────────────────────────────────────────────────────

        # ─── stop stopwatch & accumulate by table name ─────────────────────────────
        elapsed = time.perf_counter() - start
        self._add_timing[table_name] = self._add_timing.get(table_name, 0.0) + elapsed

    def get_add_timing(self, table_name=None):
        """
        If table_name is None, returns the full dict of {table_name: total_seconds}.
        Otherwise returns the cumulative seconds for that one table (or 0.0).
        """
        if not hasattr(self, '_add_timing'):
            return {} if table_name is None else 0.0
        if table_name is None:
            return self._add_timing
        return self._add_timing.get(table_name, 0.0)



    def create_table(self, table_name, obj, context, exclude_keys):
        """
        Create a table dynamically based on the object's attributes and context fields.
        """
        # Generate the schema
        schema = self.create_schema(obj, context, exclude_keys)

        # Generate the SQL for creating the table
        columns = ", ".join([f"{name} {type}" for name, type in schema.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
        #print(f"Creating table ==>\n{sql}")
        # Execute the table creation
        self.cursor.execute(sql)

        # Track the schema
        self.tables[table_name] = schema

    def create_schema(self, obj, context, exclude_keys):
        """
        Generate the schema for a table, prioritizing context fields before object fields.
        """
        # Infer schema for context fields
        context_schema = {}
        for key, value in context.items():
            if isinstance(value, int):
                context_schema[key] = "INTEGER"
            elif isinstance(value, float):
                context_schema[key] = "REAL"
            elif isinstance(value, str):
                context_schema[key] = "TEXT"
            else:
                raise TypeError(f"Unsupported context field type: {type(value)} for '{key}'")

        # Infer schema for object fields
        object_schema = self._infer_schema(obj, exclude_keys)

        # Merge context and object schemas, with context first
        return {**context_schema, **object_schema}

    def executeorig(self, sql):
        """
        Execute a SQL command that doesn't return a result set.
        """
        try:
            self.cursor.execute(sql)
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL execution failed: {e}")

    def execute(self, sql, table_name=None):
        """
        Execute a SQL command that doesn't return a result set.
        If table_name is given, track cumulative execution time per table.
        """
        # start timer if we're tracking this table
        if table_name:
            #if not hasattr(self, '_add_timing'):
            #    self._add_timing = {}
            start = time.perf_counter()
        try:
            self.cursor.execute(sql)
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL execution failed: {e}")
        else:
            # on successful run, record elapsed time
            if table_name:
                elapsed = time.perf_counter() - start
                self._add_timing[table_name] = self._add_timing.get(table_name, 0.0) + elapsed

    def executemany(self, sql, data_list, table_name=None):
        """
        Execute a SQL command that takes multiple parameter sets.
        If table_name is given, track cumulative execution time per table.
        """
        # start timer if we're tracking this table
        if table_name:
            if not hasattr(self, '_add_timing'):
                self._add_timing = {}
            start = time.perf_counter()
        else:
            1/0 #throw error to find out what isn't being logged
        try:
            self.cursor.executemany(sql, data_list)
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL execution failed: {e}")
        else:
            # on successful run, record elapsed time
            if table_name:
                elapsed = time.perf_counter() - start
                self._add_timing[table_name] = self._add_timing.get(table_name, 0.0) + elapsed

    def query(self, sql, params=None, as_dict=True):
        try:
            self.cursor.execute(sql, params or ())
            rows = self.cursor.fetchall()
            if as_dict:
                column_names = [description[0] for description in self.cursor.description]
                return [dict(zip(column_names, row)) for row in rows]
            return rows
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL query failed: {e}\nquery = {sql}") from None #TODO this should put it one up th ecall stack..

    def query_value(self, sql, params=None):
        """Return single scalar value from query. Returns None if no results."""
        rs = self.query(sql, params, as_dict=False)
        if rs and rs[0]:
            return rs[0][0]
        return None

    def query_scalar_list(self, sql, params=None):
        """
        Fetch a single-column result as a flat list.

        This method ensures that if a query selects only one column,
        the result is returned as a list of scalar values instead of tuples.

        Raises:
            RuntimeError: If the SQL query selects more than one column.
        """
        try:
            self.cursor.execute(sql, params or ())
            if len(self.cursor.description) > 1:
                raise RuntimeError("query_scalar_list() can only be used for single-column queries.")

            rows = self.cursor.fetchall()
            return [row[0] for row in rows] if rows else []
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL query failed: {e}")

    def query_print(self, sql, as_dict=True, use_excel=False, print_source = True):
        data = self.query(sql, as_dict=as_dict)  # Fetch the data from the database

        if not data:
            print(f"No results found. ==>{sql}")
            return

        if use_excel:
            headers = []
            rows = []

            if as_dict:
                headers = list(data[0].keys())
                rows = [list(row.values()) for row in data]
            else:
                headers = [f"Col_{i}" for i in range(len(data[0]))]
                rows = [list(row) for row in data]  # Unpack tuple into list

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            log_folder = Path("../..") / "gladiator_matches/prints"
            log_folder.mkdir(parents=True, exist_ok=True)
            filename = log_folder / f"QueryExport_{timestamp}.csv"

            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(rows)

            print(f"✅ Exported {len(rows)} rows to {filename}")
            os.startfile(filename)

        else:
            #from tabulate import tabulate
            report = "temp" #tabulate(data, headers="keys" if as_dict else [], tablefmt="fancy_grid")
            if print_source:
                print(f"PRINTING FROM RamDB query_print: {self.get_call_stack_line()}")
            print(report)

        return data

    def list_tables(self, detail_level=2):
        """
        :param detail_level - 1 just tables.  2 tables and fields, 3 include data type
        Dynamically list all tables in the database using sqlite_master.
        """
        if detail_level not in (1, 2, 3):
            raise RuntimeError(f"Invalid value for detail_level (use 1, 2, or 3), not ==> {detail_level}")

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]

        if not table_names:
            print("No tables found.")
            return

        if detail_level == 1:
            print("fixme") # tabulate([{"Table Name": name} for name in table_names], headers="keys", tablefmt="fancy_grid"))

        elif detail_level == 2:
            all_fields = {}
            for name in table_names:
                cursor.execute(f"PRAGMA table_info({name})")
                columns = [row[1] for row in cursor.fetchall()]
                all_fields[name] = columns
            print("fixme") # tabulate(all_fields, headers="keys", tablefmt="fancy_grid"))

        elif detail_level == 3:
            for name in table_names:
                cursor.execute(f"PRAGMA table_info({name})")
                schema = [{"Column": row[1], "Type": row[2]} for row in cursor.fetchall()]
                print(f"\nTable: {name}")
                print("fixme") # tabulate(schema, headers="keys", tablefmt="fancy_grid"))

    def get_call_stack_line(self):
        """
        Print the call stack (excluding this function itself) on one line:
          proc3 << proc2 << proc1
        """
        # grab the current stack; [0] is this frame, so skip it
        frames = inspect.stack()[1:]
        # extract function names, skipping the top-level module
        names = [frame.function for frame in frames if frame.function != '<module>']
        if names:
            return " << ".join(names)
        else:
            return "(no callers)"



    def copy_tables_to_permanent(self, db_name='arena_history.db', subfolder='history'):
        """
        Copy all tables from the in-memory SQLite database (self.conn) into the
        permanent SQLite database on disk (overwriting any existing tables with the same name).

        Skips the internal 'sqlite_sequence' table.

        Parameters:
        - db_name (str): filename of the permanent DB (defaults to 'arena_history.db').
        - subfolder (str): name of the folder under the grandparent directory where the permanent DB lives.
        """
        # 1. Open (or create) the permanent database file.
        perm_conn = self.get_perm_db_connection(db_name=db_name, subfolder=subfolder)
        perm_cursor = perm_conn.cursor()

        # 2. Get a cursor on the in-memory DB.
        mem_cursor = self.conn.cursor()

        # 3. Enumerate all tables in-memory (including their CREATE statements).
        mem_cursor.execute(
            "SELECT name, sql "
            "FROM sqlite_master "
            "WHERE type='table'"
        )
        tables = mem_cursor.fetchall()  # List of (table_name, create_sql)

        for name, create_sql in tables:
            # Skip SQLite's internal sequence table.
            if name == 'sqlite_sequence':
                continue

            # 4a. Drop the table if it already exists in the permanent DB.
            drop_sql = f"DROP TABLE IF EXISTS {name}"
            perm_cursor.execute(drop_sql)
            #print(f"drop_sql= {drop_sql}")

            # 4b. Re-create the table exactly as it was in-memory.
            #    (The 'sql' column already contains something like
            #     "CREATE TABLE table_name (col1 INTEGER, col2 TEXT, ...)" )
            perm_cursor.execute(create_sql)
            #print(f"create_sql= {create_sql}")

            # 4c. Copy rows from in-memory -> permanent.
            # 4c. Copy rows from in-memory -> permanent (chunked for speed & memory safety)
            mem_cursor.execute(f"SELECT * FROM {name} LIMIT 1")
            sample_row = mem_cursor.fetchone()

            if sample_row:
                num_cols = len(sample_row)
                placeholders = ", ".join("?" for _ in range(num_cols))
                insert_sql = f"INSERT INTO {name} VALUES ({placeholders})"

                CHUNK_SIZE = 1000
                buffer = []

                for row in mem_cursor.execute(f"SELECT * FROM {name}"):
                    buffer.append(row)
                    if len(buffer) >= CHUNK_SIZE:
                        perm_cursor.executemany(insert_sql, buffer)
                        buffer.clear()

                # Final flush
                if buffer:
                    perm_cursor.executemany(insert_sql, buffer)



            """  REMOVE IF ABOVE CODE IS WORKING BETTER>>> PERHAPS SPEED TEST mem_cursor.execute(f"SELECT * FROM {name}")
            rows = mem_cursor.fetchall()
            if rows:
                # Build a placeholder string like "?, ?, ?, …" matching the column count
                placeholders = ", ".join("?" for _ in rows[0])
                perm_cursor.executemany(
                    f"INSERT INTO {name} VALUES ({placeholders})",
                    rows
                )
            """
        # 5. Commit & close the permanent connection
        perm_conn.commit()
        perm_conn.close()

    def get_perm_db_connection(self, db_name='arena_history.db', subfolder='history'):
        """
        Connects to an SQLite database located in the specified subfolder within the parent directory of this script.
        If the subfolder does not exist, it is created.

        Parameters:
        - db_name (str): Name of the database file.
        - subfolder (str): Name of the subfolder where the database file is located.

        Returns:
        - conn: SQLite3 connection object.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up 3 levels: engine -> NNA -> src -> project_root
        project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
        subfolder_path = os.path.join(project_root, subfolder)

        try:
            os.makedirs(subfolder_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {subfolder_path}: {e}")
            raise
        db_path = os.path.join(subfolder_path, db_name)
        try:
            conn = sqlite3.connect(db_path)
            return conn
        except sqlite3.Error as e:
            print(f"Error connecting to database at {db_path}: {e}")
            raise


'''
class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """

    def __init__(self, nid: int, input_count: int, learning_rate: float, layer_id: int = 0):
        self.nid = nid
        self.layer_id = layer_id  # Add layer_id to identify which layer the neuron belongs to
        self.input_count = input_count
        self.weights = np.array([(nid + 1) * 0 for _ in range(input_count)], dtype=np.float64)

Sample usage
N1 = Neuron(1, 1, .001, 1)
N2 = Neuron(2, 2, .001, 1)
N3 = Neuron(3, 3, .001, 2)

db = RamDB()
# Add a neuron with context
db.add(N1, epoch=1, iteration=10)
db.add(N2, epoch=1, iteration=20)
db.add(N3, epoch=2, iteration=30)
result = db.query("SELECT * FROM Neuron")
result2 = db.query("SELECT * FROM Neuron", False)
print(result)
print(result2)
'''


