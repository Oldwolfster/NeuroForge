# DisplayPanelInput.py

import json
import pygame

from src.NNA.utils.general_text import smart_format
from src.NeuroForge import Const
from src.NeuroForge.EZFormLEFT import EZForm



class DisplayPanelInput(EZForm):
    """Panel displaying input values for the current sample."""

    def __init__(self, width_pct: int, height_pct: int, left_pct: int, top_pct: int,
                 bg_color=Const.COLOR_WHITE, banner_color=Const.COLOR_BLUE, hover_popup=None):
        training_data = Const.TRIs[0].training_data
        input_labels = training_data.feature_labels[:-1]  # All but target

        fields = {label: "0.000" for label in input_labels}

        super().__init__(
            fields=fields,
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            banner_text="Inputs",
            banner_color=banner_color,
            bg_color=bg_color,
            font_color=Const.COLOR_BLACK,
            same_line=True,
            hover_popup=hover_popup
        )

    def update_me(self):
        rs = Const.dm.get_sample_data(Const.TRIs[0].run_id)
        inputs = self.parse_inputs(rs.get("inputs_unscaled", "[]"))

        for i, label in enumerate(self.fields.keys()):
            if i < len(inputs):
                self.fields[label] = smart_format(float(inputs[i]))
            else:
                self.fields[label] = "N/A"

    def parse_inputs(self, raw_inputs):
        """Parse JSON input string to list."""
        try:
            return json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
        except json.JSONDecodeError:
            return []


