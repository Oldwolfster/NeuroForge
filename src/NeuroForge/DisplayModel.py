# DisplayModel.py

import pygame

from src.NNA.utils.general_text import format_percent
from src.NeuroForge import Const
from src.NeuroForge.ButtonBase import Button_Base
from src.NeuroForge.DisplayArrow import DisplayArrow
from src.NeuroForge.DisplayModel__Graph import DisplayModel__Graph
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.GeneratorNeuron import GeneratorNeuron
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from src.NeuroForge.PopupArchitecture import ArchitecturePopup
from src.NNA.utils.general_text import beautify_text


class DisplayModel(EZSurface):
    def __init__(self, TRI: TrainingRunInfo, position: dict):
        super().__init__(width_pct=0, height_pct=0, left_pct=0, top_pct=0,pixel_adjust_width=position["width"],pixel_adjust_height=position["height"],pixel_adjust_left=position["left"],pixel_adjust_top=position["top"],bg_color=Const.COLOR_FOR_BACKGROUND)
        self.TRI                        = TRI
        self.run_id                     = TRI.run_id
        self.config                     = TRI.config
        self.last_epoch                 = TRI.last_epoch
        self.neurons                    = []
        self.layers                     = []
        self.buttons                    = []
        self.hoverlings                 = []
        self.arch_popup                 = ArchitecturePopup(self, self.config)
        self.graph                      = None
        self.graph_holder               = None
        self.layer_width                = 0
        self.input_scaler_neuron        = None
        self.prediction_scaler_neuron   = None
        self.thresholder                = None
        self.needs_arrow_rebuild        = False
        btn                             = Button_Base(
                text                    =self.get_model_button_text(),
                width_pct               =10, height_pct=4, left_pct=1, top_pct=1,
                on_click                =self.show_info,
                on_hover                =lambda: self.arch_popup.show_me(),
                shadow_offset           =-5, auto_size=True, my_surface=self.surface,
                border_radius           =8,
                text_line2              =f"Accuracy: {format_percent(TRI.best_accuracy)} ",
                surface_offset          =(self.left, self.top))

        self.buttons.append(btn)
        #self.hoverlings.append(btn)
        Const.dm.hoverlings.append(btn)
    def show_info(self):#onclick model button
        print("copy to clipboard")
    def get_model_button_text(self):
        """
        Determines the button text for this model.
        Only adds run_id number if multiple models share the same optimizer name.
        """
        optimizer_counts = {}
        for tri in Const.TRIs:
            name = tri.gladiator
            optimizer_counts[name] = optimizer_counts.get(name, 0) + 1

        # Only add number if there are multiple models with this optimizer name
        needs_number = optimizer_counts[self.TRI.gladiator] > 1
        return f"{beautify_text(self.TRI.gladiator)} {self.TRI.run_id}" if needs_number else beautify_text(
            self.TRI.gladiator)

    def initialize_with_model_info(self):
        """Create neurons based on architecture."""
        max_activation = self.get_max_activation()
        GeneratorNeuron.create_neurons(self, max_activation)
        self.graph = self.create_graph(self.graph_holder)  # Add Graph  # MAE over epoch

        if self.thresholder:              Const.dm.hoverlings.append(self.input_scaler_neuron)
        if self.input_scaler_neuron:      Const.dm.hoverlings.append(self.input_scaler_neuron)
        if self.prediction_scaler_neuron: Const.dm.hoverlings.append(self.prediction_scaler_neuron)

        #if self.input_scaler_neuron:      self.hoverlings.append(self.input_scaler_neuron)
        #if self.prediction_scaler_neuron: self.hoverlings.append(self.prediction_scaler_neuron)





        Const.dm.eventors.append(self.graph)
        # Register neurons as hoverlings
        for layer in self.neurons:
            for neuron in layer:
                neuron.model = self
                # handled in layerself.hoverlings.append(neuron)

    def create_output_to_prediction_scaler_arrow(self):
        # Arrow: output neuron → prediction scaler (always needed, even without thresholder)

        if self.prediction_scaler_neuron and self.prediction_scaler_neuron.neuron_visualizer.my_fcking_labels:
            output_neuron = self.neurons[-1][0]  # Last layer, first neuron

            # Start: right side of output neuron, near bottom
            start_x = output_neuron.location_right_side - 3.69
            start_y = output_neuron.location_bottom_side + 3.69

            # End: left side of prediction scaler's "Prediction" oval
            end_x = self.prediction_scaler_neuron.neuron_visualizer.my_fcking_labels[0][0]
            end_y = self.prediction_scaler_neuron.neuron_visualizer.my_fcking_labels[0][1]

            DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface, arrow_size=16, thickness=2).draw()

    def create_output_to_thresholder(self):
        """Draw arrows: output neuron → prediction scaler → thresholder → branches"""
        if not self.thresholder:
            return

        # Safety check: wait for first render to populate coordinates
        if not self.prediction_scaler_neuron.neuron_visualizer.my_fcking_labels:
            return

        from src.NeuroForge.DisplayArrow import DisplayArrow

        # Arrow: prediction scaler → thresholder diamond (left side)
        # Coordinates are already in local model space - don't add self.left/self.top
        start_x, start_y = self.prediction_scaler_neuron.neuron_visualizer.label_y_positions[1]
        end_x, end_y = self.thresholder.neuron_visualizer.diamond_left

        DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface, arrow_size=16, thickness=2).draw()

        # Arrow: diamond top → NO branch (alpha/min)
        start_x, start_y = self.thresholder.neuron_visualizer.diamond_top
        end_x, end_y = self.thresholder.neuron_visualizer.arrow_targets[0]

        DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface, arrow_size=16, thickness=2,
                     label_text="NO").draw()

        # Arrow: diamond right → corner (no arrow head)
        start_x, start_y = self.thresholder.neuron_visualizer.diamond_right
        end_x, end_y = self.thresholder.neuron_visualizer.arrow_targets[1]

        DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface, arrow_size=0, thickness=2).draw()

        # Arrow: corner → YES branch (beta/max)
        start_x, start_y = self.thresholder.neuron_visualizer.arrow_targets[1]
        end_x, end_y = self.thresholder.neuron_visualizer.arrow_targets[2]

        DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface, arrow_size=16, thickness=2,
                     label_text="YES").draw()
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
        if self.graph is not None:
            self.graph.render()
        # Draw scaler neurons if present
        if self.input_scaler_neuron:
            self.input_scaler_neuron.draw_neuron()
        if self.prediction_scaler_neuron:
            self.prediction_scaler_neuron.draw_neuron()
            self.create_output_to_prediction_scaler_arrow()
        if self.thresholder:
            self.thresholder.draw_neuron()
            self.create_output_to_thresholder()

        # Draw all neurons
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron()
        for button in self.buttons:            button.draw_me()
        self.render_internal_arrows()
        self.render_scaler_to_hidden_arrows()


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



    def render_scaler_to_hidden_arrows(self):
        """Draw arrows from input scaler to first hidden layer, fresh each frame."""
        from src.NeuroForge.DisplayArrow import DisplayArrow

        if not self.input_scaler_neuron:
            return

        first_layer = self.neurons[0]
        scaler_positions = self.input_scaler_neuron.neuron_visualizer.label_y_positions

        # Tunable offsets
        start_y_offset = 16
        end_y_offset = 8

        for neuron in first_layer:
            if not neuron.on_screen:
                continue

            labels = neuron.neuron_visualizer.my_fcking_labels

            for input_index, (start_x, start_y) in enumerate(scaler_positions):
                if input_index + 1 >= len(labels):
                    continue

                end_x = labels[input_index + 1][0]
                end_y = labels[input_index + 1][1] + end_y_offset

                DisplayArrow(start_x, start_y + start_y_offset, end_x, end_y, screen=self.surface).draw()
    def update_me(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron()

    def draw_border(self):
        pygame.draw.rect(
            self.surface, Const.COLOR_FOR_NEURON_BODY,
            (0, 0, self.width, self.height), 3
        )

    def create_graph(self, gh):
        doublewide = gh.location_width * 2 + 20
        return DisplayModel__Graph(left=gh.location_left, width=doublewide, top=gh.location_top,
                                   height=gh.location_height, model_surface=self.surface, run_id=self.run_id,
                                   model=self)
