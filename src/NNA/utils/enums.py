from enum import Enum


class RecordLevel(Enum):
    NONE    = 0     # No recording — e.g., hyperparameter sweep (LR probe)
    SUMMARY = 1     # Basic stats: accuracy, loss, convergence, etc.
    FULL    = 2     # + Iteration history, weight deltas, etc. (NeuroForge playback)
    DEBUG   = 3     # + Diagnostics, blame signals, and dev-level traces