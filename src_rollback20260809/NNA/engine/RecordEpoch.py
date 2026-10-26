from dataclasses import dataclass

@dataclass
class RecordEpoch:
    run_id:     int
    epoch:      int
    phase:      int
    correct:    int
    wrong:      int
    accuracy:   float
    mae:        float