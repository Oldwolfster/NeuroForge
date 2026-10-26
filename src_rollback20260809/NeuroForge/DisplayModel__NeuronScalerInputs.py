# DisplayModel__NeuronScalerInputs.py
import pygame

from src.NNA.utils.general_text import smart_format
from src.NeuroForge import Const
import json


class DisplayModel__NeuronScalerInputs:
    """
    Strategy for input scaler visualization.
    Each row shows: raw_value | ScalerType | scaled_value (all on same pill)

    Owns layout; calls dispatcher primitives for drawing.
    """

    # Layout constants - percentages/tweakable
    ROW_HEIGHT = 19
    ROW_SPACING = 1.69
    TOP_OFFSET = 46
    OVERHANG = 44
    HEIGHT_PADDING = 0

    def __init__(self, neuron):
        self.neuron = neuron
        self.scale_methods = neuron.config.scaler.get_scaling_names()

        self.adjust_placement()

    def adjust_placement(self):
        self.original_top = self.neuron.location_top
        self.neuron.location_width = self.neuron.location_width *.69
        row_count=self.count_rows()
        if row_count>8: self.ROW_SPACING = 1.15
        self.neuron.location_top = self.original_top + self.TOP_OFFSET
        self.neuron.location_height = self.TOP_OFFSET + row_count * self.ROW_HEIGHT * self.ROW_SPACING + self.HEIGHT_PADDING


    def count_rows(self):
        rs = Const.dm.get_sample_data(self.neuron.model.run_id)
        unscaled = self.parse_list(rs.get("inputs_unscaled", "[]"))
        return len(unscaled)

    def render(self):
        """Render all input rows"""
        self.neuron.draw_top_plane(y_offset=-10)

        rows = self.get_rows()
        for index, row_data in enumerate(rows):
            self.draw_row(index, row_data)
        self.neuron.need_label_coord = False

    def get_rows(self):
        """Return [(label, raw_value, scaled_value), ...] for each input feature"""
        rs = Const.dm.get_sample_data(self.neuron.model.run_id)
        unscaled = self.parse_list(rs.get("inputs_unscaled", "[]"))
        scaled = self.parse_list(rs.get("inputs", "[]"))

        rows = []
        for i, (raw, scl) in enumerate(zip(unscaled, scaled)):
            label = self.scale_methods[i] if i < len(self.scale_methods) else ""
            rows.append((label, raw, scl))
        return rows

    def draw_row(self, index, row_data):
        """Draw three-part pill: raw | label | scaled"""
        label, raw_value, scaled_value = row_data

        y_pos = self.TOP_OFFSET + index * self.ROW_HEIGHT * self.ROW_SPACING + self.neuron.location_top
        x = self.neuron.location_left - self.OVERHANG
        width = self.neuron.location_width + self.OVERHANG * 2

        # Draw pill
        pill_rect = (x, y_pos, width, self.ROW_HEIGHT)
        self.neuron.draw_pill(pill_rect)

        # Schedule text draws (global coords)
        global_pill = self.neuron.to_global_rect(pygame.Rect(pill_rect))
        global_pill.y += 2  # vertical tweak

        self.neuron.schedule_text(global_pill, smart_format(raw_value), Const.COLOR_WHITE, 'left')
        self.neuron.schedule_text(global_pill, label, Const.COLOR_WHITE, 'center')
        self.neuron.schedule_text(global_pill, smart_format(scaled_value), Const.COLOR_WHITE, 'right')

        # Store arrow anchors
        if self.neuron.need_label_coord:
            self.store_arrow_anchors(y_pos)

    def store_arrow_anchors(self, y_pos):
        """Store arrow anchor coordinates for connection drawing"""
        center_y_IN = y_pos + self.ROW_HEIGHT * 0.5
        center_y_OUT = y_pos - self.ROW_HEIGHT * 0.5
        left_x = self.neuron.location_left - self.OVERHANG
        right_x = self.neuron.location_left + self.neuron.location_width + self.OVERHANG

        self.neuron.my_fcking_labels.append((left_x, center_y_IN))
        self.neuron.label_y_positions.append((right_x, center_y_OUT))

    def parse_list(self, raw):
        """Parse JSON string or pass through list/scalar"""
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, (int, float)):
            return [raw]
        return []

    # Proxy properties for arrow coord access via old path
    @property
    def my_fcking_labels(self):
        return self.neuron.my_fcking_labels

    @property
    def label_y_positions(self):
        return self.neuron.label_y_positions