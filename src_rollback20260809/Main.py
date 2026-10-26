from logging import Logger

from src.ArenaSettings import HyperParameters
from src.NNA.TriFromDisk import TriFromDisk
from src.NNA.engine.NeuroEngine import NeuroEngine
from src.NNA.utils.db_conn import db_conn
from src.NeuroForge.NeuroForge import NeuroForge
from src.NNA.shard.ShardDispatcher import ShardDispatcher

def run_original():
    """No args = dev mode. Runs from ArenaSettings.py as always."""
    #Logger.log      ( 1,"*** NNA PATH — no input scalers resolved. Debug only; not model comparison. ***")
    shared_hyper    = HyperParameters()
    engine          = NeuroEngine(shared_hyper)
    engine          . run_a_batch()
    if engine.TRIs  : NeuroForge(engine.TRIs)


def spoof():
    import sys
    #sys.argv = ["main.py", "--mode", "create_db",      "--db", r"C:\Users\silian\.neuroforge\projects\Test.nf"]
    #sys.argv = ["main.py", "--mode", "prepare_batch",  "--db", r"C:\Users\silian\.neuroforge\projects\Test.nf"]
    #sys.argv = ["main.py", "--mode", "run_batch",      "--db", r"C:\Users\silian\.neuroforge\projects\Test.nf", "--shard", "0,7"]
    pass


if __name__ == '__main__':
    import sys

    #spoof()
    if len(sys.argv) > 1:       ShardDispatcher().dispatch()
    else:                       run_original()

    #db_conn.open_project("NSFromTable")


    #NeuroForge([TriFromDisk.latest(db_conn.open_project()  )])        # db = your db_conn, no NNA run needed