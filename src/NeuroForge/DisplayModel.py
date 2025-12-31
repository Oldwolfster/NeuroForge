# DisplayModel.py

import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.GeneratorNeuron import GeneratorNeuron
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo


class DisplayModel(EZSurface):
    def __init__(self, TRI: TrainingRunInfo, position: dict):
        super().__init__(
            width_pct=0, height_pct=0, left_pct=0, top_pct=0,
            pixel_adjust_width=position["width"],
            pixel_adjust_height=position["height"],
            pixel_adjust_left=position["left"],
            pixel_adjust_top=position["top"],
            bg_color=Const.COLOR_FOR_BACKGROUND
        )
        self.TRI = TRI
        self.run_id = TRI.run_id
        self.config = TRI.config
        self.last_epoch = TRI.last_epoch
        self.neurons = []
        self.layers = []
        self.hoverlings = []
        self.graph = None
        self.graph_holder = None
        self.layer_width = 0
        self.input_scaler_neuron = None
        self.prediction_scaler_neuron = None
        self.thresholder = None
        self.needs_arrow_rebuild = False

    def initialize_with_model_info(self):
        """Create neurons based on architecture."""
        max_activation = self.get_max_activation()
        GeneratorNeuron.create_neurons(self, max_activation)

        # Register neurons as hoverlings
        for layer in self.neurons:
            for neuron in layer:
                neuron.model = self
                self.hoverlings.append(neuron)

    def get_max_activation(self):
        """Get 95th percentile max activation for scaling."""
        sql = """
            SELECT MAX(abs_activation) AS max_activation
            FROM (
                SELECT ABS(activation_value) AS abs_activation
                FROM Neuron
                WHERE run_id = ?
                ORDER BY abs_activation ASC
                LIMIT (SELECT CAST(COUNT(*) * 0.95 AS INT) 
                       FROM Neuron WHERE run_id = ?)
            )
        """
        result = self.TRI.db.query(sql, (self.run_id, self.run_id))
        return result[0]['max_activation'] if result and result[0]['max_activation'] else 1.0

    @property
    def display_epoch(self):
        """Returns epoch to display, capped at model's last epoch if converged early."""
        if self.last_epoch is None:
            return Const.vcr.CUR_EPOCH
        return min(Const.vcr.CUR_EPOCH, self.last_epoch)

    def render(self):
        self.clear()
        self.draw_border()

        # Draw scaler neurons if present
        if self.input_scaler_neuron:
            self.input_scaler_neuron.draw_neuron()
        if self.prediction_scaler_neuron:
            self.prediction_scaler_neuron.draw_neuron()

        # Draw all neurons
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron()

        self.render_internal_arrows()

    # DisplayModel.py

    def render_internal_arrows(self):
        """Draw arrows between neurons within this model, recreated fresh each frame."""
        from src.NeuroForge.DisplayArrow import DisplayArrow

        # Tunable offsets
        start_y_offset = +6.9  # From bottom of neuron (negative = up)
        end_x_offset = -5  # Move left into gap between neurons
        end_y_offset = 8  # Move down to center of label

        for layer_idx in range(len(self.neurons) - 1):
            source_layer = self.neurons[layer_idx]
            dest_layer = self.neurons[layer_idx + 1]

            for src_neuron in source_layer:
                if not src_neuron.on_screen:
                    continue

                # Arrow start: right side of source neuron, near bottom
                start_x = src_neuron.location_right_side -3.69
                start_y = src_neuron.location_bottom_side + start_y_offset

                for dst_neuron in dest_layer:
                    if not dst_neuron.on_screen:
                        continue

                    # Weight index = src_neuron.position + 1 (bias is weight[0])
                    weight_idx = src_neuron.position + 1

                    visualizer = getattr(dst_neuron, 'neuron_visualizer', None)
                    if visualizer and weight_idx < len(visualizer.my_fcking_labels):
                        label_x, label_y = visualizer.my_fcking_labels[weight_idx]
                        end_x = label_x + end_x_offset
                        end_y = label_y + end_y_offset
                        DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface).draw()

    def update_me(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron()

    def draw_border(self):
        pygame.draw.rect(
            self.surface, Const.COLOR_FOR_NEURON_BODY,
            (0, 0, self.width, self.height), 3
        )