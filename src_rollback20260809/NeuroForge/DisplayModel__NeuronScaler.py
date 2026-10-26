import pygame
from src.NNA.utils.pygame import draw_gradient_rect
from src.NNA.utils.general_text import smart_format
from src.NeuroForge.DisplayModel__NeuronScalerInputs import DisplayModel__NeuronScalerInputs
from src.NeuroForge.DisplayModel__NeuronScalerPrediction import DisplayModel__NeuronScalerPrediction
from src.NeuroForge.DisplayModel__NeuronScalerThresholder import DisplayModel__NeuronScalerThresholder
from src.NeuroForge.DisplayModel__Neuron_Base import DisplayModel__Neuron_Base
from src.NeuroForge.PopupInputScaler import PopupInputScaler
from src.NeuroForge.PopupPredictionScaler import PopupPredictionScaler
from src.NeuroForge import Const


class DisplayModel__NeuronScaler(DisplayModel__Neuron_Base):
    """
    Dispatcher for scaler visualizations. Owns rendering primitives.
    Strategies own their layout and call these primitives.

    Strategies:
        NeuronScalerInputs      - N rows showing raw → scaled for each input
        NeuronScalerPrediction  - 3 rows showing scaled → unscaled for output
        NeuronScalerThresholder - Diamond decision node (fully autonomous)
    """

    BANNER_HEIGHT = 40

    def _from_base_constructor(self):
        """Called from DisplayModel_Neuron_Base constructor"""
        self.on_screen = True
        self.my_fcking_labels = []
        self.label_y_positions = []
        self.need_label_coord = True
        self.is_thresholder = (self.nid == -2)

        self.apply_layout_adjustments()
        self.create_strategy()
        self.font = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)

    def apply_layout_adjustments(self):
        """Normalize dimensions and position"""
        target_width = 136.9
        if self.location_width > target_width:
            if not self.is_input:
                self.location_left += (self.location_width - target_width) * 0.5
            self.location_width = target_width

    def create_strategy(self):
        """Instantiate appropriate visualizer strategy and popup"""
        if self.is_input:
            self.banner_text = "Scaler"
            self.neuron_visualizer = DisplayModel__NeuronScalerInputs(self)
            self.popup = PopupInputScaler(self)

        elif self.is_thresholder:
            self.banner_text = "Thresholder"
            self.neuron_visualizer = DisplayModel__NeuronScalerThresholder(self, self.ez_printer)

        else:
            self.banner_text = "Prediction"
            self.position_right_of_output_neuron()
            self.neuron_visualizer = DisplayModel__NeuronScalerPrediction(self)
            self.popup = PopupPredictionScaler(self)

    def position_right_of_output_neuron(self):
        """Place prediction scaler to the right of the last neuron"""
        last_neuron = self.model.neurons[-1][-1]
        self.location_left = last_neuron.location_left + last_neuron.location_width + 30

    def render_tooltip(self):
        """Render popup when hovered"""
        if hasattr(self, 'popup') and self.popup:
            self.popup.show_me()

    def draw_neuron(self):
        """Draw the neuron visualization"""
        if self.is_thresholder:
            self.neuron_visualizer.render()
            return

        self.neuron_visualizer.render()
        self.draw_body()
        self.draw_banner()

    def draw_body(self):
        """Draw the neuron body outline"""
        font = pygame.font.Font(None, 30)
        label_surface = font.render(self.banner_text, True, Const.COLOR_FOR_NEURON_TEXT)
        label_strip_height = label_surface.get_height() + 8

        body_y = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        body_rect = (self.location_left, body_y, self.location_width, body_height)
        pygame.draw.rect(self.screen, Const.COLOR_FOR_NEURON_BODY, body_rect, border_radius=6, width=7)

    def draw_banner(self):
        """Draw the gradient banner header"""
        font = pygame.font.Font(None, 30)
        label_surface = font.render(self.banner_text, True, Const.COLOR_FOR_NEURON_TEXT)
        label_strip_height = label_surface.get_height() + 8

        banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        draw_gradient_rect(self.screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END,
                           border_radius=6)

        text_x = self.location_left + 5
        text_y = self.location_top + 5 + (label_strip_height - label_surface.get_height()) // 2
        self.screen.blit(label_surface, (text_x, text_y))

    def draw_top_plane(self, y_offset=-10):
        """Draw header pill behind banner"""
        rect = pygame.Rect(self.location_left, self.location_top + y_offset, self.location_width, self.BANNER_HEIGHT)
        self.draw_pill(rect)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMITIVES - Strategies call these
    # ─────────────────────────────────────────────────────────────────────────

    def draw_pill(self, rect, color=Const.COLOR_BLUE):
        """Draw horizontal pill (rectangle with circular end caps)"""
        x, y, w, h = rect
        radius = h // 2

        center_rect = pygame.Rect(x + radius, y, w - 2 * radius, h)
        pygame.draw.rect(self.screen, color, center_rect)
        pygame.draw.circle(self.screen, color, (x + radius, y + radius), radius)
        pygame.draw.circle(self.screen, color, (x + w - radius, y + radius), radius)

    def blit_text_aligned(self, surface, area_rect, text, font, color, align, padding=5):
        """Render text with specified alignment within area"""
        surf = font.render(str(text), True, color)
        r = surf.get_rect()
        r.centery = area_rect.centery

        if align == 'left':
            r.x = area_rect.x + padding
        elif align == 'right':
            r.right = area_rect.right - padding
        else:
            r.centerx = area_rect.centerx

        surface.blit(surf, r)

    def schedule_text(self, rect, text, color, align, padding=8):
        """Schedule text draw for later (global coords)"""
        Const.dm.schedule_draw(self.blit_text_aligned, Const.SCREEN, rect, text, self.font, color, align, padding)

    def to_global_coords(self, x, y):
        """Convert local neuron coordinates to global screen coordinates"""
        return (x + self.model.left, y + self.model.top)

    def to_global_rect(self, rect):
        """Convert local rect to global screen coordinates"""
        return rect.move(self.model.left, self.model.top)