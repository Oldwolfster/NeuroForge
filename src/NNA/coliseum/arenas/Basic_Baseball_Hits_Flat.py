from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple


class Basic_Baseball_Hits_Flat(BaseArena):
    """
    Four games of a hitter's box score plus how many at-bats he gets tomorrow.
    Only the two most recent games feed the answer; the older two are there to be
    ignored, along with every home run and walk column.

    Lesson:       Recency. Twelve of the seventeen inputs are dead weight.
    Truth:        Hits5 = round(0.6 x G4 Hits + 0.2 x G3 Hits + 0.2 if G4 HR else 0
                  + uniform(-0.25, 0.25)), clamped to 0..Game 5 AB
    Solvable by:  A single neuron, once it zeroes the dead columns.
    Watch for:    Game 4 H carrying roughly three times the weight of Game 3 H.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, ...]]:
        training_data = []
        for _ in range(self.num_samples):
            sequence = []
            for _ in range(4):
                ab = random.randint(2, 6)                  # at-bats
                h = random.randint(0, ab)                  # hits ≤ AB
                hr = random.randint(0, min(2, h))          # homers ≤ hits
                bb = random.randint(0, 3)                  # walks
                sequence.append((ab, h, hr, bb))

            # Fifth game AB (random)
            ab5 = random.randint(2, 6)

            # --- Dependency Rule for H5 ---
            recent_hits = sequence[-1][1] * 0.6 + sequence[-2][1] * 0.2
            bonus_if_hr_yesterday = 0.2 if sequence[-1][2] > 0 else 0.0
            raw_target = recent_hits + bonus_if_hr_yesterday + random.uniform(-0.25, 0.25)
            h5 = max(0, min(ab5, round(raw_target)))  # clamp to valid range

            # Flatten all features + target
            features = [x for game in sequence for x in game]
            features.append(ab5)
            features.append(h5)
            training_data.append(tuple(features))
        return training_data, [  # NEW — dedent this whole block 8 spaces
            "Game 1 AB", "Game 1 H", "Game 1 HR", "Game 1 BB",
            "Game 2 AB", "Game 2 H", "Game 2 HR", "Game 2 BB",
            "Game 3 AB", "Game 3 H", "Game 3 HR", "Game 3 BB",
            "Game 4 AB", "Game 4 H", "Game 4 HR", "Game 4 BB",
            "Game 5 AB",
            "Game 5 H"
        ]
