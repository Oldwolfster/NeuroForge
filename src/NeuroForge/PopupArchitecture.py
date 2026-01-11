from typing import List

import pygame

from src.NNA.utils.general_text import smart_format, beautify_text
from src.NeuroForge import Const
from src.NeuroForge.Popup_Base import Popup_Base
from src.NNA.engine.Config import Config



class ArchitecturePopup(Popup_Base):
    def __init__(self, model, configs: List[Config]):
        super().__init__(
            column_width_overrides={0: 30}
        )


    def header_text(self):
        return "Model Architecture Comparison"


    def draw_dividers(self, surf, col_w):

        y = self.y_coord_for_row(1)
        #pygame.draw.line(surf, Const.COLOR_BLACK, (0,y),(ARCH_W,y),2)
    # no highlights required here


    def content_to_display(self):
        configs = [tri.config for tri in Const.TRIs]

        def get_labels():
            label_rows = [
                ("Model Definition",     ""),  # spacer
                ("",                     "Neuron Layout"),
                ("",                     "Initializer"),
                ("",                     "Hidden Activation"),
                ("",                     "Output Activation"),
                ("",                     "Loss Function :("),
                ("",                     ""),  # spacer

                ("Training Setup",       ""),
                ("",                     "Optimizer"),
                ("",                     "Batch Size"),
                ("",                     "Learning Rate"),
                ("",                     "Input Scalers")
            ]
            # Add per-input + target scaler labels
            if configs and hasattr(configs[0], "scaler"):
                first_scaler = configs[0].scaler
                for label in first_scaler.get_scaling_labels():
                    label_rows.append(("", label))

            label_rows += [
                ("",                     "ROI Mode"),
                ("",                     ""),  # spacer

                ("Training Outcome",     ""),
                ("",                     "Training Time (s)"),
                ("",                     "Final Epoch"),
                ("",                     "Best Error"),
                ("",                     "Best Error @ Epoch"),
                ("",                     "Convergence"),
            ]
            return  label_rows

        def architecture(bp) -> str:

            if bp == [1]:
                return "Perceptron"
            else:
                layers = ", ".join(str(x) for x in bp)
                return f"{layers} Neurons/Layer"


        def hidden_activation (arch, hid_act) -> str:

            if arch == [1]:
                return "No Hidden Neurons"
            else:

                return hid_act

        def describe(TRI):
            cfg = TRI.config
            describe_rows = [
                self.get_model_name(TRI),
                architecture(cfg.architecture),
                cfg.weight_initializer.name,
                hidden_activation(cfg.architecture, cfg.hidden_activation.name),
                cfg.output_activation.name,
                cfg.loss_function.name,
                "",

                "",  # Training Setup header, no value
                cfg.optimizer.name,
                f"{cfg.batch_size} (Samples)",
                smart_format(cfg.learning_rate),
            ]
            describe_rows.append("-------------")
            for scaling_name in cfg.scaler.get_scaling_names():
                describe_rows.append(beautify_text(scaling_name))
            describe_rows += [
                #beautify_text(cfg.roi_mode.name),
                beautify_text("Coming Soon"),
                "",

                "",  # Training Outcome header
                smart_format(TRI.time_seconds),
                f"{TRI.last_epoch} (Epochs)",
                smart_format(TRI.lowest_mae),
                TRI.lowest_mae_epoch,
                TRI.converge_cond,
            ]
            return describe_rows  # <== This was over-indented before


        # These were incorrectly inside `describe()`
        labels = get_labels()
        label_count = len(labels)

        def safe_describe(TRI):
            rows = describe(TRI)
            if len(rows) < label_count:
                rows += [""] * (label_count - len(rows))
            elif len(rows) > label_count:
                rows = rows[:label_count]
            return rows
        #configs = [tri.config for tri in Const.TRIs]
        rows = [[group, label] + [safe_describe(TRI)[i] for TRI in Const.TRIs] for i, (group, label) in enumerate(labels)]
        #rows = [[group, label] + [safe_describe(cfg)[i] for cfg in configs] for i, (group, label) in enumerate(labels)]
        return [list(col) for col in zip(*rows)]


    def get_model_name(self, TRI):
        """Get model name with numbering if duplicates exist."""
        optimizer_counts = {}
        for tri in Const.TRIs:
            name = tri.gladiator
            optimizer_counts[name] = optimizer_counts.get(name, 0) + 1

        needs_number = optimizer_counts[TRI.gladiator] > 1
        return f"{beautify_text(TRI.gladiator)} {TRI.run_id}" if needs_number else beautify_text(TRI.gladiator)
