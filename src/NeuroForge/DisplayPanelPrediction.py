import pygame

from src.NNA.utils.general_text import smart_format
from src.NeuroForge import Const
from src.NeuroForge.EZFormLEFT import EZForm


class DisplayPanelPrediction(EZForm):
    __slots__ = ("run_id", "problem_type", "loss_function", "target_name")

    def __init__(self, run_id: int, problem_type: str, TRI,
                 width_pct: int, height_pct: int, left_pct: int, top_pct: int):

        self.run_id         = run_id
        self.problem_type   = problem_type
        self.loss_function  = TRI.config.loss_function
        self.target_name    = TRI.training_data.feature_labels[-1].strip()

        fields = {
            "Sample Error": "0.000",
            "Epoch Avg Err": "0.000",
            "Loss Function": self.loss_function.short_name,
            f"{self.loss_function.short_name} Gradient": "0.000",
        }

        super().__init__(
            fields=fields,
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            banner_text="Error Stats",
            banner_color=Const.COLOR_BLUE
        )

    def update_me(self):
        rs_sample = Const.dm.get_sample_data(self.run_id)
        rs_epoch  = Const.dm.get_epoch_data(self.run_id)

        if not rs_sample:
            return  # Gracefully handle missing sample

        error         = rs_sample.get("error_unscaled", 0.0)
        loss_gradient = rs_sample.get("loss_gradient", 0.0)
        is_true       = rs_sample.get("is_true")

        avg_error = rs_epoch.get("mean_absolute_error_unscaled", 0.0) if rs_epoch else 0.0

        # Binary Decision banner feedback
        if self.problem_type == "Binary Decision":
            if is_true is True:
                self.banner_text = "Correct"
                self.set_colors(1)
            elif is_true is False:
                self.banner_text = "Wrong"
                self.set_colors(0)

        self.fields["Sample Error"] = smart_format(error)
        self.fields["Epoch Avg Err"] = smart_format(avg_error)
        self.fields[f"{self.loss_function.short_name} Gradient"] = smart_format(loss_gradient)
