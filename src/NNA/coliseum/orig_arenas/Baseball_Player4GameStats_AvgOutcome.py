from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple


class Player4GameStats_AvgOutcome(BaseArena):
    """
    Simulates a baseball player's 4-game stat history (ABs and Hits per game)
    and predicts their expected batting average in the next game.

    Inputs:
    - AB1, H1, AB2, H2, AB3, H3, AB4, H4

    Output:
    - Projected batting average next game (0.000 to 1.000)

    You can easily switch this to OBP or SLG later.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, ...]], List[str]]:
        training_data = []
        for _ in range(self.num_samples):
            game_stats = []
            total_hits = 0
            total_abs = 0

            for _ in range(4):
                ab = random.randint(2, 6)                         # Typical range per game
                hits = min(ab, random.randint(0, ab))             # Can't have more hits than ABs
                game_stats.extend([ab, hits])
                total_abs += ab
                total_hits += hits

            # Add some noise and clamp to [0, 1]
            projected_avg = total_hits / total_abs if total_abs > 0 else 0
            projected_avg += random.gauss(0, 0.02)
            projected_avg = max(0.0, min(1.0, projected_avg))

            training_data.append((*game_stats, projected_avg))

        return training_data, [
            "Game 1 AB", "Game 1 Hits",
            "Game 2 AB", "Game 2 Hits",
            "Game 3 AB", "Game 3 Hits",
            "Game 4 AB", "Game 4 Hits",
            "Projected Next Game Batting Avg"
        ]