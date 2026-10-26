# RunIdentity.py                                    NEW FILE
# Lives beside DisplayModel.py in NeuroForge.
from src.NNA.utils.general_text  import smart_format, format_percent
from src.NNA.engine.TrainingRunInfo import TEST
from src.NeuroForge import Const


class RunIdentity:
    """
    Answers one question: what makes this run THIS run?
    The comparison set is whatever NeuroScope loaded -- nothing else.
    Banner renders the answer as rows, popup joins it into a header.
    Neither one decides what is interesting, so they cannot drift.
    """

    def __init__(self, TRI, peers):
        self.TRI    = TRI
        self.peers  = peers

    # ---------------------------------------------------------------- priority

    def priority_fields(self):
        """Reorder this list to change what wins the mid row. That is the whole knob."""
        return [
            ("Arena",             lambda TRI: TRI.arena),
            ("Optimizer",         lambda TRI: TRI.config.optimizer.name),
            ("Learning Rate",     lambda TRI: smart_format(TRI.config.learning_rate)),
            ("Neuron Layout",     lambda TRI: str(TRI.config.architecture)),
            ("Hidden Activation", lambda TRI: TRI.config.hidden_activation.name),
            ("Output Activation", lambda TRI: TRI.config.output_activation.name),
            ("Loss Function",     lambda TRI: TRI.config.loss_function.name),
            ("Initializer",       lambda TRI: TRI.config.weight_initializer.name),
            ("Batch Size",        lambda TRI: str(TRI.config.batch_size)),
            ("Target Scaler",     lambda TRI: TRI.config.target_scaler.name),
            ("Input Scalers",     lambda TRI: ", ".join(TRI.config.scaler.get_scaling_names())),
        ]

    # ---------------------------------------------------------------- the diff

    def value(self, fn, TRI):
        """A field that cannot answer is skipped as an identity -- never a crash mid-render."""
        try:                return str(fn(TRI))
        except Exception:   return None

    def differs(self, fn) -> bool:
        """A field only separates runs if every peer could answer AND they disagree."""
        seen = {self.value(fn, tri) for tri in self.peers}
        return len(seen) > 1 and None not in seen

    def diffs(self):
        """Priority order, capped at three. Four differences is noise, not identity."""
        found = [(label, self.value(fn, self.TRI))
                 for label, fn in self.priority_fields() if self.differs(fn)]
        return found[:3]

    def field(self, wanted: str):
        """Fallback reads through the same list, so it can never name a field that died."""
        for label, fn in self.priority_fields():
            if label == wanted: return self.value(fn, self.TRI)
        return None

    # ---------------------------------------------------------------- renderings

    def line_top(self) -> str:
        return f"Seed {self.TRI.seed}    Run {self.TRI.run_id}"

    def line_mid(self) -> str:
        found = self.diffs()
        if not found:           return self.context()
        if len(found) == 1:     return f"{found[0][0]}: {found[0][1]}"
        return ", ".join(value for label, value in found)

    def line_bottom(self) -> str:
        score = self.TRI.best_accuracy[TEST]
        if score is None:   return "Accuracy: --"
        return f"Accuracy: {format_percent(score)}"

    def context(self) -> str:
        """Nothing separates these runs -- say what they ARE instead. Slash, so it reads as context."""
        parts = [self.field("Arena"), self.field("Optimizer")]
        return " / ".join(part for part in parts if part)

    def header(self) -> str:
        """Popup wants one string where the banner wants three rows. Same answer, folded."""
        return f"Run {self.TRI.run_id}  {self.line_mid()}"


class deleteme:

    # DisplayModel.py  method: get_model_button_text  Update: identity rows replace the gladiator name
    def get_model_button_text(self):
        return RunIdentity(self.TRI, Const.TRIs).line_top()


    # DisplayModel.py  import block  Update
    from src.NeuroForge.RunIdentity import RunIdentity                          # NEW

    # PopupArchitecture.py  method: get_model_name  Update: identity, no gladiator
    def get_model_name(self, TRI) -> str:
        return RunIdentity(TRI, Const.TRIs).header()

