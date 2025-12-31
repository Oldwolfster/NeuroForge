# DisplayPanelTarget.py

import pygame

from src.NNA.utils.general_text import smart_format
from src.NeuroForge import Const
from src.NeuroForge.EZFormLEFT import EZForm



class DisplayPanelTarget(EZForm):
    """Panel displaying the target value for the current sample."""

    def __init__(self, width_pct: int, height_pct: int, left_pct: int, top_pct: int,
                 bg_color=Const.COLOR_WHITE, banner_color=Const.COLOR_BLUE):
        training_data = Const.TRIs[0].training_data
        self.target_name = training_data.feature_labels[-1]

        fields = {self.target_name: ""}

        super().__init__(
            fields=fields,
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            banner_text="Target",
            banner_color=banner_color,
            bg_color=bg_color,
            font_color=Const.COLOR_BLACK
        )

    def update_me(self):
        rs = Const.dm.get_sample_data(Const.TRIs[0].run_id)
        target = rs.get("target_unscaled", "")
        display_value = smart_format(target)

        # Add label if binary decision
        training_data = Const.TRIs[0].training_data
        if training_data.is_binary_decision:
            label = self.get_target_label(target, training_data)
            if label:
                display_value = f"{display_value} - {label}"

        self.fields[self.target_name] = display_value

    def get_target_label(self, target, training_data):
        """Return the class label for a binary decision target value."""
        BD = Const.TRIs[0].BD
        if not BD.is_active:
            return None
        try:
            return BD.label_max if float(target) == BD.target_max else BD.label_min
        except (ValueError, TypeError):
            return None