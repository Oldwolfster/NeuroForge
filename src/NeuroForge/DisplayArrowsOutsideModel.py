from src.NeuroForge import Const
from src.NeuroForge.DisplayArrow import DisplayArrow

class DisplayArrowsOutsideNeuron:
    """Static arrows connecting panels to model edges. Built once, rendered each frame."""

    # Tuning constants (percentages of screen dimensions)
    INPUT_ARROW_X_OFFSET = 0.010        # ~19px at 1920
    INPUT_ARROW_Y_OFFSET = 0.004        # ~4px at 1080
    OUTPUT_ARROW_Y_OFFSET = 0.006       # ~6px at 1080
    GRADIENT_ARROW_LENGTH = 0.042       # ~80px at 1920

    def __init__(self, model, is_top: bool):
        self.model = model
        self.is_top = is_top
        self.arrows = []
        self.built = False


    def draw_me(self):
        if not self.built:
            if self.is_top:
                self.build_input_arrows()
            self.build_output_arrows()
            self.built = True
        for arrow in self.arrows:
            arrow.draw()

    def build_input_arrows(self):
        """Input panel → Input Scaler (top model only)"""
        input_panel = Const.dm.input_panel
        scaler = self.model.input_scaler_neuron
        if not scaler:
            return

        visualizer = scaler.neuron_visualizer
        x_offset = Const.SCREEN_WIDTH * self.INPUT_ARROW_X_OFFSET
        y_offset = Const.SCREEN_HEIGHT * self.INPUT_ARROW_Y_OFFSET

        for i, (start_x, start_y) in enumerate(input_panel.label_y_positions):
            if i >= len(visualizer.my_fcking_labels):
                break
            end_x = self.model.left + visualizer.my_fcking_labels[i][0]
            end_y = self.model.top + visualizer.my_fcking_labels[i][1] + y_offset
            self.arrows.append(DisplayArrow(start_x - x_offset, start_y, end_x, end_y, screen=Const.SCREEN))

    # DisplayArrowsOutsideNeuron.py

    def build_output_arrows(self):
        """Prediction Scaler → Error panel + gradient stub back"""
        prediction_panel = self.get_prediction_panel()
        if not prediction_panel:
            return

        scaler = self.model.prediction_scaler_neuron
        if not scaler:
            return

        y_offset = Const.SCREEN_HEIGHT * self.OUTPUT_ARROW_Y_OFFSET
        visualizer = scaler.neuron_visualizer

        # Forward arrow: scaler → Sample Error field
        start_x = self.model.left + visualizer.label_y_positions[2][0]
        start_y = self.model.top + visualizer.label_y_positions[2][1]
        end_x = prediction_panel.left  # LEFT edge of panel
        end_y = prediction_panel.label_y_positions[0][1]  # Y from field position

        self.arrows.append(DisplayArrow(
            start_x, start_y, end_x, end_y - y_offset,
            screen=Const.SCREEN,
            thickness=6,
            arrow_size=50,
            layer_colors=((255, 0, 0), (0, 255, 0), (0, 0, 255))
        ))

        # Gradient arrow: horizontal stub pointing left from gradient field
        grad_y = prediction_panel.label_y_positions[3][1]
        grad_x = prediction_panel.left  # Start at LEFT edge of panel
        arrow_length = Const.SCREEN_WIDTH * self.GRADIENT_ARROW_LENGTH

        self.arrows.append(DisplayArrow(
            grad_x, grad_y, grad_x - arrow_length, grad_y,
            screen=Const.SCREEN,
            thickness=4,
            arrow_size=40,
            layer_colors=((255, 0, 0), (0, 255, 0), (0, 0, 255))
        ))
    def get_prediction_panel(self):
        """Find the prediction panel matching this model's run_id."""
        from src.NeuroForge.DisplayPanelPrediction import DisplayPanelPrediction
        for comp in Const.dm.components:
            if isinstance(comp, DisplayPanelPrediction) and comp.run_id == self.model.run_id:
                return comp
        return None


    def update_me(self):
        pass