
# ═══ src/NNA/arenas/Basic_Income_1Input_NoBias.py  (NEW — complete file) ═══════
# The docstring below is the standard: untagged summary first, then any of
#   Lesson: / Truth: / Source: / Solvable by: / Watch for: / Notes:
# and any other  Label:  line you care to add. All optional. The About pane read    s them.
from src.NNA.engine.BaseArena import BaseArena
import random


class Basic_Income_1Input_NoBias(BaseArena):
    """
    Income from years of experience and nothing else. Every year is worth $4,000
    and a rookie earns $0, so the line runs straight through the origin.

    Lesson:       One weight and no bias — the simplest thing that can learn.
    Truth:        Salary = 4000 × Years Exp + noise(σ 5000)
    Solvable by:  A single neuron. The bias is not needed.
    Watch for:    The weight should settle near 4000 and the bias should stay near 0.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self):
        rows = []
        for _ in range(self.num_samples):
            years  = random.uniform(0, 40)
            salary = max(0.0, 4000 * years + random.gauss(0, 5000))
            rows.append((years, salary))
        return rows, ["Years Exp", "Salary"]