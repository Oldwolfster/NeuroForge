
import os
import sqlite3
from pathlib     import Path
from collections import namedtuple

ProjectInfo = namedtuple("ProjectInfo", [
    "name",             # display name, no .nf
    "path",             # full Path
    "mtime",            # last modified
    "run_count",        # runs completed
    "expected_runs",    # runs expected across all batches
    "batch_count",      # number of batches
])


class db_conn:
    """The .nf files: where they live, which is active, and a connection to one.

       Classmethods own the folder.  Instances own a single open connection.
    """

    PROJECTS_FOLDER     = Path.home() / ".neuroforge" / "projects"
    active_project_path = None

    # ══════════════════════════════════════════════════════════════════════════
    #   PUBLIC GATEWAY
    #   - open_project (will create if needed)
    #   - run (all statements go through here?

    # ══════════════════════════════════════════════════════════════════════════
    @classmethod
    def open_project(cls, name=None):
        return cls.ensure_schema(cls.resolve_path(name))

    def run(self, sql, params=None, many=False):
        try:
            if many: self.cursor.executemany(sql, params)
            else:    self.cursor.execute(sql, params or ())
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL FAILED: {e}\n----- SQL -----\n{sql}\n---------------") from e




    # db_conn.py  method: ensure_folder  NEW:  projects folder exists
    @classmethod
    def ensure_folder(cls):
        cls.PROJECTS_FOLDER.mkdir(parents=True, exist_ok=True)

    # db_conn.py  method: get_db_path  NEW:  active project, or the default
    @classmethod
    def get_db_path(cls):
        if cls.active_project_path: return cls.active_project_path
        return cls.get_default_project()

    # db_conn.py  method: set_active_project  NEW:  picker calls this
    @classmethod
    def set_active_project(cls, path):
        cls.active_project_path = Path(path) if path else None

    # db_conn.py  method: get_default_project  NEW:  most recent, else create one
    @classmethod
    def get_default_project(cls):
        projects = cls.list_projects()
        if projects: return projects[0].path
        return cls.create_project("default")

    # db_conn.py  method: create_project  NEW:  was a subprocess, now a call
    @classmethod
    def create_projectOld(cls, name):
        from src.NNA.utils.db_schema import db_schema        # lazy: db_schema imports db_conn
        path = cls.next_free_path(name)
        db   = db_schema.create_database(path)
        db.close()
        return path

    # db_conn.py  method: next_free_path  NEW:  timestamp-suffix on collision
    @classmethod
    def next_free_path(cls, name):
        from datetime import datetime
        cls.ensure_folder()
        safe_name = cls.sanitize_filename(name)
        path      = cls.PROJECTS_FOLDER / f"{safe_name}.nf"
        if not path.exists(): return path
        stamp     = datetime.now().strftime("%Y_%m_%d_%H_%M")
        return cls.PROJECTS_FOLDER / f"{safe_name}_{stamp}.nf"

    # db_conn.py  method: list_projects  NEW:  ProjectInfo list, newest first
    @classmethod
    def list_projects(cls):
        cls.ensure_folder()
        projects = [cls.get_project_info(p) for p in cls.PROJECTS_FOLDER.glob("*.nf")]
        projects = [p for p in projects if p]
        return sorted(projects, key=lambda p: p.mtime, reverse=True)

    # db_conn.py  method: get_project_info  NEW:  mtime from disk, counts from db
    @classmethod
    def get_project_info(cls, path):
        try:
            run_count, expected_runs, batch_count = cls.query_project_counts(path)
            return ProjectInfo(
                name          = path.stem,
                path          = path,
                mtime         = os.path.getmtime(path),
                run_count     = run_count,
                expected_runs = expected_runs,
                batch_count   = batch_count)
        except Exception:
            return None

    # db_conn.py  method: query_project_counts  NEW:  brief open, three counts
    @classmethod
    def query_project_counts(cls, path):
        conn = sqlite3.connect(path)
        try:
            cur           = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM batch_history")
            run_count     = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*), COALESCE(SUM(expected_runs), 0) FROM batch")
            batch_count, expected_runs = cur.fetchone()
            return run_count, expected_runs, batch_count
        except sqlite3.OperationalError:
            return 0, 0, 0
        finally:
            conn.close()

    # db_conn.py  method: sanitize_filename  NEW:  spaces to underscores, strip unsafe
    @classmethod
    def sanitize_filename(cls, name):
        result = name.strip()
        for char in '<>:"/\\|?*': result = result.replace(char, '_')
        return result.replace(' ', '_') or "untitled"

    # db_conn.py  method: delete_project  NEW:  UI confirms first
    @classmethod
    def delete_project(cls, path):
        Path(path).unlink()

    # db_conn.py  method: rename_project  NEW:  returns the new path
    @classmethod
    def rename_project(cls, old_path, new_name):
        safe_name = cls.sanitize_filename(new_name)
        new_path  = cls.PROJECTS_FOLDER / f"{safe_name}.nf"
        if new_path.exists(): raise FileExistsError(f"Project '{safe_name}' already exists")
        Path(old_path).rename(new_path)
        return new_path

    # ══════════════════════════════════════════════════════════════════════════
    #   CONNECTION LEVEL
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self, db_path=None):
        self.db_path    = str(db_path or db_conn.get_db_path())
        self.conn       = sqlite3.connect(self.db_path, isolation_level=None)
        self.cursor     = self.conn.cursor()
        self.apply_pragmas()

    # db_conn.py  method: apply_pragmas  NEW:  future-state pragmas, once per connection
    def apply_pragmas(self):
        self.cursor.execute("PRAGMA journal_mode = WAL")      # readers never block the writer
        self.cursor.execute("PRAGMA synchronous  = NORMAL")   # safe under app crash, fast
        self.cursor.execute("PRAGMA busy_timeout = 5000")     # wait for the lock, don't throw
        self.cursor.execute("PRAGMA temp_store   = MEMORY")

    # db_conn.py  method: begin / commit / rollback / close  NEW:  explicit transactions
    def begin(self):        self.cursor.execute("BEGIN IMMEDIATE")
    def commit(self):       self.cursor.execute("COMMIT")
    def rollback(self):     self.cursor.execute("ROLLBACK")
    def close(self):        self.conn.close()

    def execute(self, sql, params=None):
        self.run(sql, params)

    def executemany(self, sql, rows):
        self.run(sql, rows, many=True)

    def query(self, sql, params=None):
        self.run(sql, params)
        columns = [d[0] for d in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def query_rows(self, sql, params=None):
        self.run(sql, params)
        return self.cursor.fetchall()

    def query_value(self, sql, params=None):
        self.run(sql, params)
        row = self.cursor.fetchone()
        return row[0] if row else None

    def query_column(self, sql, params=None):
        self.run(sql, params)
        return [row[0] for row in self.cursor.fetchall()]

    def query_scalar_list(self, sql, params=None):
        self.run(sql, params)
        if len(self.cursor.description) > 1:
            raise RuntimeError(f"query_scalar_list expects ONE column, got "
                               f"{len(self.cursor.description)}: {sql}")
        return [row[0] for row in self.cursor.fetchall()]


    # db_conn.py  method: resolve_path  NEW:  name -> path.  None means active project
    @classmethod
    def resolve_path(cls, name=None):
        if name is None:            return Path(cls.get_db_path())
        path = Path(name)
        if path.suffix == ".nf":    return path
        cls.ensure_folder()
        return cls.PROJECTS_FOLDER / f"{cls.sanitize_filename(name)}.nf"

    # db_conn.py  method: ensure_schema  NEW:  idempotent — the only db_schema caller
    @classmethod
    def ensure_schema(cls, path):
        from src.NNA.utils.db_schema import db_schema     # lazy: db_schema imports db_conn
        return db_schema.create_database(path)

    # db_conn.py  method: create_project  Update:  UI new-project, shares ensure_schema
    @classmethod
    def create_project(cls, name):
        path = cls.next_free_path(name)
        db   = cls.ensure_schema(path)
        db.close()
        return path

