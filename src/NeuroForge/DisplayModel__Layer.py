import pygame
from src.NeuroForge import Const

class DisplayModel__Layer:
    """Manages a single layer's neurons, zoom state, and pagination."""

    MIN_HEIGHT_STANDARD = 91
    MIN_HEIGHT_COMPACT = 10

    def __init__(self, model, layer_index, x_position, width, available_height):
        self.model = model
        self.layer_index = layer_index
        self.x_position = x_position
        self.width = width
        self.available_height = available_height

        self.neurons = []
        self.needs_controls = False
        self.control_mode = "scroll"  # "scroll" or "zoom"
        self.target_height = self.MIN_HEIGHT_STANDARD
        self.current_offset = 0
        self.visible_count = 0

        self.control_buttons = []
        self.scroll_up_button = None
        self.scroll_down_button = None
        self.zoom_button = None

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def determine_control_needs(self):
        """Called after neurons added - decide if scrolling/zoom needed."""
        if not self.neurons:
            return False

        actual_height = self.neurons[0].location_height
        total_layers = len(self.model.config.architecture)
        is_output_layer = (self.layer_index == total_layers - 1)
        is_comfortable = (actual_height >= self.MIN_HEIGHT_STANDARD)

        self.needs_controls = (not is_output_layer) and (not is_comfortable)
        print(f"Layer {self.layer_index}: height={actual_height:.1f}, comfortable={is_comfortable}, output={is_output_layer}, needs_controls={self.needs_controls}")

        if self.needs_controls:
            self.current_offset = 0
            self.recalculate_visible_count()
            self.create_control_buttons()
            self.apply_visibility()
            self.reposition_visible_neurons()

        # Mark all neurons visible if no controls needed
        if not self.needs_controls:
            for neuron in self.neurons:
                neuron.on_screen = True

        return self.needs_controls

    def recalculate_visible_count(self):
        """Calculate how many neurons fit at current target_height."""
        usable = self.available_height - Const.NEURON_MIN_GAP
        self.visible_count = max(1, int(usable / (self.target_height + Const.NEURON_MIN_GAP)))

    def apply_visibility(self):
        """Mark neurons visible based on current offset and count."""
        for i, neuron in enumerate(self.neurons):
            neuron.on_screen = (self.current_offset <= i < self.current_offset + self.visible_count)

    def reposition_visible_neurons(self):
        """Calculate positions for visible neurons."""
        if self.visible_count < 1:
            return

        total_neuron_height = self.visible_count * self.target_height
        if self.visible_count > 1:
            gap = (self.available_height - total_neuron_height) / (self.visible_count + 1)
        else:
            gap = (self.available_height - self.target_height) / 2.0

        visible_index = 0
        for neuron in self.neurons:
            if neuron.on_screen:
                y_pos = visible_index * self.target_height + ((visible_index + 1) * gap)
                neuron.location_top = y_pos
                neuron.location_height = self.target_height
                neuron.location_bottom_side = y_pos + self.target_height

                if hasattr(neuron, 'neuron_visualizer') and neuron.neuron_visualizer:
                    neuron.neuron_visualizer.recalculate_layout()
                visible_index += 1

    def update_layout(self):
        """Refresh layout after scroll/zoom change."""
        self.apply_visibility()
        self.reposition_visible_neurons()
        self.model.needs_arrow_rebuild = True
        if self.layer_index == 0:
            Const.dm.needs_outside_arrow_rebuild = True

    def clamp_offset(self):
        """Keep offset within valid bounds."""
        max_offset = max(0, len(self.neurons) - self.visible_count)
        self.current_offset = max(0, min(self.current_offset, max_offset))

    # ─────────────────────────────────────────────────────────────
    # Button Actions
    # ─────────────────────────────────────────────────────────────

    def scroll_up(self):
        """+ button: scroll up or zoom in."""
        if not self.needs_controls:
            return

        if self.control_mode == "scroll":
            if self.current_offset > 0:
                self.current_offset -= 1
                self.update_layout()
        else:
            self.target_height += 10
            self.recalculate_visible_count()
            self.clamp_offset()
            self.update_layout()

    def scroll_down(self):
        """- button: scroll down or zoom out."""
        if not self.needs_controls:
            return

        if self.control_mode == "scroll":
            if self.current_offset + self.visible_count < len(self.neurons):
                self.current_offset += 1
                self.update_layout()
        else:
            if self.target_height > self.MIN_HEIGHT_COMPACT:
                self.target_height -= 10
                self.recalculate_visible_count()
                self.clamp_offset()
                self.update_layout()

    def toggle_zoom(self):
        """Switch between scroll and zoom modes."""
        if not self.needs_controls:
            return

        if self.control_mode == "scroll":
            self.control_mode = "zoom"
            self.zoom_button.text = "Zoom"
            self.zoom_button.text_line2 = "Scroll"
        else:
            self.control_mode = "scroll"
            self.zoom_button.text = "Scroll"
            self.zoom_button.text_line2 = "Zoom"

    # ─────────────────────────────────────────────────────────────
    # Button Creation
    # ─────────────────────────────────────────────────────────────

    def create_control_buttons(self):
        """Create +/Scroll-Zoom/- button bar."""
        from src.NeuroForge.ButtonBase import Button_Base

        if self.control_buttons:
            print(f"Layer {self.layer_index}: buttons already exist, skipping")
            return

        print(f"Layer {self.layer_index}: creating control buttons at model.top={self.model.top}")

        screen = Const.SCREEN
        screen_width, screen_height = screen.get_size()

        scroll_width = 30
        zoom_width = 70
        button_height = 28
        gap = 4
        bar_width = 2 * scroll_width + gap + zoom_width

        global_left = self.model.left + self.x_position
        start_x = int(global_left + (self.width - bar_width) / 2)
        y = int(self.model.top + 10)

        def make_button(x, w, text, on_click, font_size=20, text_line2=None, text_line2_color=None):
            btn = Button_Base(
                my_surface=screen,
                text=text,
                width_pct=(w / screen_width) * 100,
                height_pct=(button_height / screen_height) * 100,
                left_pct=(x / screen_width) * 100,
                top_pct=(y / screen_height) * 100,
                on_click=on_click,
                shadow_offset=-3,
                font_size=font_size,
                padding=4,
                text_line2=text_line2,
                text_line2_color=text_line2_color,
                text_line2_offset=-5 if text_line2 else 0
            )
            self.control_buttons.append(btn)
            return btn

        self.scroll_up_button = make_button(start_x, scroll_width, "+", self.scroll_up, 22)

        zoom_x = start_x + scroll_width + gap
        self.zoom_button = make_button(zoom_x, zoom_width, "Scroll", self.toggle_zoom, 18,
                                       text_line2="Zoom", text_line2_color=Const.COLOR_GRAY)

        minus_x = zoom_x + zoom_width + gap
        self.scroll_down_button = make_button(minus_x, scroll_width, "-", self.scroll_down, 22)