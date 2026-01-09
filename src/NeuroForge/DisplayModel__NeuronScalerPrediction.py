# DisplayModel__NeuronScalerPrediction.py

import pygame
from src.NNA.utils.general_text import smart_format
from src.NeuroForge import Const


class DisplayModel__NeuronScalerPrediction:
    """
    Strategy for prediction scaler visualization.

    Two modes:
        Scaling active: Three-part pills like InputScaler (scaled | label | unscaled)
        No scaling:     Label rows + value rows, right-aligned numbers, no pills

    Owns layout; calls dispatcher primitives for drawing.
    """

    # Layout constants
    ROW_HEIGHT = 19
    ROW_SPACING = 1.15
    TOP_OFFSET = 35
    OVERHANG = 24
    HEIGHT_PADDING = 459

    def __init__(self, neuron):
        self.neuron = neuron
        output_nid  = neuron.nid-1
        #self.neuron.location_top = self.neuron.model.neurons[output_nid].location_top
        if self.has_scaling():
            self.ROW_SPACING = 1.69
        else:
            self.HEIGHT_PADDING = 3
            self.neuron.location_width = self.neuron.location_width *.8


    def render(self):
        """Render prediction/target/error rows"""
        self.neuron.draw_top_plane(y_offset=-7)

        if self.has_scaling():
            self.render_scaled_mode()
        else:
            self.render_unscaled_mode()



        self.neuron.need_label_coord = False

    def has_scaling(self):
        """Check if target scaling is active"""
        rs = Const.dm.get_sample_data(self.neuron.run_id)
        return (rs.get("prediction_raw") != rs.get("prediction_unscaled") or
                rs.get("target") != rs.get("target_unscaled") or
                rs.get("error") != rs.get("error_unscaled"))

    def get_scaled_rows(self):
        """Return [(label, scaled, unscaled), ...] for scaled mode"""
        rs = Const.dm.get_sample_data(self.neuron.run_id)
        return [
            ("Target", rs.get("target", ""), rs.get("target_unscaled", "")),
            ("Prediction", rs.get("prediction_raw", ""), rs.get("prediction_unscaled", "")),
            ("Error", rs.get("error", ""), rs.get("error_unscaled", "")),
        ]

    def get_unscaled_rows(self):
        """Return [(label, value), ...] for unscaled mode - 6 rows total"""
        rs = Const.dm.get_sample_data(self.neuron.run_id)
        return [
            ("Target", None),
            (None, rs.get("target", "")),
            ("Prediction", None),
            (None, rs.get("prediction_raw", "")),
            ("Error", None),
            (None, rs.get("error", "")),
        ]

    def render_scaled_mode(self):
        """Render three-part pills with overhang"""
        rows = self.get_scaled_rows()

        for index, row_data in enumerate(rows):
            self.draw_scaled_row(index, row_data)

        self.neuron.location_height = self.TOP_OFFSET + len(
            rows) * self.ROW_HEIGHT * self.ROW_SPACING + self.HEIGHT_PADDING

    def render_unscaled_mode(self):
        """Render label/value pairs without pills"""
        rows = self.get_unscaled_rows()

        for index, row_data in enumerate(rows):
            self.draw_unscaled_row(index, row_data)

        self.neuron.location_height = self.TOP_OFFSET + len(
            rows) * self.ROW_HEIGHT * self.ROW_SPACING + self.HEIGHT_PADDING

    def draw_scaled_row(self, index, row_data):
        """Draw three-part pill: scaled | label | unscaled"""
        label, scaled_value, unscaled_value = row_data

        y_pos = self.TOP_OFFSET + index * self.ROW_HEIGHT * self.ROW_SPACING + self.neuron.location_top
        x = self.neuron.location_left - self.OVERHANG
        width = self.neuron.location_width + self.OVERHANG * 2

        # Draw pill
        pill_rect = (x, y_pos, width, self.ROW_HEIGHT)
        self.neuron.draw_pill(pill_rect)

        # Schedule text draws (global coords)
        global_pill = self.neuron.to_global_rect(pygame.Rect(pill_rect))
        global_pill.y += 2

        self.neuron.schedule_text(global_pill, smart_format(scaled_value), Const.COLOR_WHITE, 'left')
        self.neuron.schedule_text(global_pill, label, Const.COLOR_WHITE, 'center')
        self.neuron.schedule_text(global_pill, smart_format(unscaled_value), Const.COLOR_WHITE, 'right')

        # Store arrow anchors
        if self.neuron.need_label_coord:
            self.store_arrow_anchors_scaled(index, label, y_pos)

    def draw_unscaled_row(self, index, row_data):
        """Draw label (left) or value (right) - no pill"""
        label, value = row_data

        y_pos = self.TOP_OFFSET + index * self.ROW_HEIGHT * self.ROW_SPACING + self.neuron.location_top

        row_rect = pygame.Rect(self.neuron.location_left, y_pos, self.neuron.location_width, self.ROW_HEIGHT)
        global_rect = self.neuron.to_global_rect(row_rect)
        global_rect.y += 2

        if label is not None:
            self.neuron.schedule_text(global_rect, label, Const.COLOR_BLACK, 'left')
        if value is not None:
            self.neuron.schedule_text(global_rect, smart_format(value), Const.COLOR_BLACK, 'right')

            # Store arrow anchors on value rows only
            if self.neuron.need_label_coord:
                self.store_arrow_anchors_unscaled(index, y_pos)

    def store_arrow_anchors_scaled(self, index, label, y_pos):
        """Store arrow anchors for scaled mode"""
        center_y = y_pos + self.ROW_HEIGHT * 0.5
        right_x = self.neuron.location_left + self.neuron.location_width + self.OVERHANG

        self.neuron.label_y_positions.append((right_x, center_y))
        if label == "Prediction":
            self.neuron.my_fcking_labels.append((self.neuron.location_left - self.OVERHANG, center_y))

    def store_arrow_anchors_unscaled(self, index, y_pos):
        """Store arrow anchors for unscaled mode (value rows only)"""
        center_y = y_pos + self.ROW_HEIGHT * 0.5
        right_x = self.neuron.location_left + self.neuron.location_width

        self.neuron.label_y_positions.append((right_x, center_y))
        # Prediction is index 3 (the value row after "Prediction" label)
        if index == 3:
            self.neuron.my_fcking_labels.append((self.neuron.location_left, center_y))

    # Proxy properties for arrow coord access via old path
    @property
    def my_fcking_labels(self):
        return self.neuron.my_fcking_labels

    @property
    def label_y_positions(self):
        return self.neuron.label_y_positions