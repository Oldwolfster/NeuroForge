import pygame
from src.NeuroForge import Const


class DisplayModel__NeuronWeightsSmall:
    """Compact 3-line neuron display for zoomed-out view"""

    def __init__(self, neuron, ez_printer):
        self.neuron = neuron
        self.ez_printer = ez_printer

        # Needed so create_neuron_to_neuron_arrows doesn't explode
        self.my_fcking_labels = []       # List of (x, y) points for arrows
        self.need_label_coord = True     # Only compute once per neuron

    def render(self):
        """Draw just the essentials in 3 lines"""
        n = self.neuron
        if not n.on_screen:
            return

        # Ensure we have label coords for arrows, even in compact mode
        if self.need_label_coord:
            self._init_label_coords()

        # Skip if neuron is too short even for compact
        if n.location_height < 25:
            return

        # Simple text, all same color for now
        y = n.location_top + 5
        x = n.location_left + 5

        # Use smaller font for compact mode
        small_font = pygame.font.Font(None, 16)

        # Line 1: Neuron ID
        text = f"{n.label}"
        text_surface = small_font.render(text, True, Const.COLOR_BLACK)
        n.screen.blit(text_surface, (x, y))

        # Line 2: Activation
        y += 10
        text = f"Act: {n.activation_value:.3f}"
        text_surface = small_font.render(text, True, Const.COLOR_BLACK)
        n.screen.blit(text_surface, (x, y))

        # Line 3: Weight Sum
        y += 10
        text = f"Sum: {n.raw_sum:.3f}"
        text_surface = small_font.render(text, True, Const.COLOR_BLACK)
        n.screen.blit(text_surface, (x, y))

    def _init_label_coords(self):
        """
        Provide at least one (x, y) per weight/input (including bias)
        so code like my_fcking_labels[input_index+1] is always in range.
        """
        n = self.neuron
        self.my_fcking_labels = []

        # Prefer weights_before (matches how compute line uses [1] + neuron_inputs)
        weights = getattr(n, "weights_before", getattr(n, "weights", []))
        num_weights = len(weights)

        # Inputs (not counting bias)
        num_inputs = len(getattr(n, "neuron_inputs", []))

        # We need enough slots to safely index [input_index + 1]
        # so: at least num_inputs + 1 (bias + each input)
        needed = max(num_weights, num_inputs + 1)+2

        if needed <= 0:
            needed = 1  # ultra-safety fallback

        step = n.location_height / (needed + 1)
        x = n.location_left + 5

        for i in range(needed):
            y = int(n.location_top + (i + 1) * step)
            self.my_fcking_labels.append((x, y))

        self.need_label_coord = False
