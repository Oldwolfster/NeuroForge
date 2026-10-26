# PopupArchitecture.py
from typing import List

from src.NNA.utils.general_text import smart_format, beautify_text, format_percent
from src.NeuroForge import Const
from src.NeuroForge.Popup_Base import Popup_Base
from src.NNA.engine.Config import Config
from src.NNA.engine.TrainingRunInfo import TRAIN, VALIDATE, TEST
from src.NeuroForge.RunIdentity import RunIdentity


class ArchitecturePopup(Popup_Base):
    def __init__(self, model, configs: List[Config]):      # TODO both args unused -- popup reads Const.TRIs
        super().__init__(
            column_width_overrides={0: 30}
        )

    def header_text(self):
        return "Model Architecture Comparison"

    # ---------------------------------------------------------------- assembly

    def content_to_display(self):
        """Each row carries its own label AND its own value, so the two can never drift apart."""
        rows = [[group, label] + [self.cell(value_of, TRI) for TRI in Const.TRIs]
                for group, label, value_of in self.row_specs()]
        return [list(col) for col in zip(*rows)]

    def cell(self, value_of, TRI) -> str:
        """Header and spacer rows declare no value. Everything else renders as text."""
        return "" if value_of is None else f"{value_of(TRI)}"

    def row_specs(self):
        return self.definition_rows() + self.setup_rows() + self.outcome_rows()

    # ---------------------------------------------------------------- the three blocks

    def definition_rows(self):
        return [
            ("Model Definition", "",                      lambda TRI: self.get_model_name(TRI)),
            ("",                 "Seed",                  lambda TRI: TRI.seed),
            ("",                 "Neuron Layout",         lambda TRI: self.architecture(TRI)),
            ("",                 "Initializer",           lambda TRI: TRI.config.weight_initializer.name),
            ("",                 "Hidden Activation",     lambda TRI: self.hidden_activation(TRI)),
            ("",                 "Output Activation",     lambda TRI: TRI.config.output_activation.name),
            ("",                 "Loss Function :(",      lambda TRI: TRI.config.loss_function.name),
            ("",                 "",                      None),
        ]

    def setup_rows(self):
        return [
            ("Training Setup",   "",                      None),
            ("",                 "Optimizer",             lambda TRI: TRI.config.optimizer.name),
            ("",                 "Batch Size",            lambda TRI: f"{TRI.config.batch_size} (Samples)"),
            ("",                 "Learning Rate",         lambda TRI: smart_format(TRI.config.learning_rate)),
            ("",                 "Input Scalers",         lambda TRI: "-------------"),
        ] + self.scaler_rows() + [
            ("",                 "ROI Mode",              lambda TRI: beautify_text("Coming Soon")),
            ("",                 "",                      None),
        ]

    def outcome_rows(self):
        return [
            ("Run Outcome",      "",                      None),
            ("",                 "Training Time (s)",     lambda TRI: smart_format(TRI.time_seconds)),
            ("",                 "Final Epoch",           lambda TRI: f"{TRI.last_epoch} (Epochs)"),
            ("",                 "MAE",                   lambda TRI: self.phase_error   (TRI, TEST)),
            ("",                 "Accuracy",              lambda TRI: self.phase_accuracy(TRI, TEST)),
            ("",                 "Convergence",           lambda TRI: TRI.converge_cond),
        ]

    # PopupArchitecture.py  method: phase_accuracy  NEW: mirrors phase_error for the accuracy array
    def phase_accuracy(self, TRI, phase: int) -> str:
        if not TRI.phase_ran(phase):    return self.phase_absent(TRI, phase)
        return  format_percent(TRI.best_accuracy[phase])

    # ---------------------------------------------------------------- scalers

    def scaler_rows(self):
        """One row per scaler slot. Slots come from model 0; each model answers for its own."""
        return [("", name, self.scaler_value(slot))
                for slot, name in enumerate(self.scaling_names(Const.TRIs[0]))]

    def scaler_value(self, slot: int):
        def value_of(TRI):
            names = self.scaling_names(TRI)
            if slot >= len(names):  return "-------------"
            return beautify_text(names[slot])
        return value_of

    def scaling_names(self, TRI):
        """One guard for labels and values both -- they can't disagree about whether a scaler exists."""
        scaler = getattr(TRI.config, "scaler", None)
        return [] if scaler is None else scaler.get_scaling_names()

    # ---------------------------------------------------------------- formatting

    def architecture(self, TRI) -> str:
        layout = TRI.config.architecture
        if layout == [1]:   return "Perceptron"
        return ", ".join(str(count) for count in layout) + " Neurons/Layer"

    def hidden_activation(self, TRI) -> str:
        if TRI.config.architecture == [1]:  return "No Hidden Neurons"
        return TRI.config.hidden_activation.name

    def phase_error(self, TRI, phase: int) -> str:
        if not TRI.phase_ran(phase):    return self.phase_absent(TRI, phase)
        return smart_format(TRI.lowest_mae[phase])

    def phase_epoch(self, TRI, phase: int) -> str:
        if not TRI.phase_ran(phase):    return self.phase_absent(TRI, phase)
        return f"{TRI.lowest_mae_epoch[phase]}"

    def phase_absent(self, TRI, phase: int) -> str:
        """Never lie about why a cell is empty: no data at all vs. data that never ran."""
        if TRI.data[phase] is None:     return "-------------"
        return "Not Run Yet"

    def get_model_name(self, TRI) -> str:
        return RunIdentity(TRI, Const.TRIs).header()