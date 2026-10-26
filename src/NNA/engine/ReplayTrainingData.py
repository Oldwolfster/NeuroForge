# ════════════════════════════════════════════════════════════════════════════
# ReplayTrainingData.py   NEW FILE  (src/NNA/engine/)
# ════════════════════════════════════════════════════════════════════════════
# ReplayTrainingData.py
import json


class ReplayTrainingData:
    """
    Stand-in for TrainingData, rebuilt from one batch_history row.
    Exposes exactly the attribute surface NeuroScope and BinaryDecision
    read from a live TrainingData -- nothing training-side.
    """
    def __init__(self, row: dict):
        self.arena_name         = row['arena']
        self.problem_type       = row['problem_type']
        self.is_binary_decision = row['problem_type'] == 'Binary Decision'
        self.sample_count       = row['sample_count']
        self.input_count        = row['input_count']
        self.feature_labels     = self.json_or_none(row['feature_labels'])
        self.source_code        = row['source_code']
        self.target_min         = row['target_min']
        self.target_max         = row['target_max']
        self.target_mean        = row['target_mean']
        self.target_stdev       = row['target_stdev']
        self.target_min_label   = row['target_min_label']
        self.target_max_label   = row['target_max_label']
        self.target_labels      = [row['target_min_label'], row['target_max_label']]

    @staticmethod
    def json_or_none(raw):
        return None if raw is None else json.loads(raw)

