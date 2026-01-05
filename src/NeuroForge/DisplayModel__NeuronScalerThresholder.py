import pygame

from src.NNA.utils.pygame import get_darker_color
from src.NeuroForge import Const


class DisplayModel__NeuronScalerThresholder:
    """
    Displays a thresholder as a decision diamond with YES/NO branches.
    Shows which branch is active and whether the prediction is correct.
    """
    # Positioning constants
    DIAMOND_OFFSET_FROM_OUTPUT = 200
    DIAMOND_RADIUS = 35
    DIAMOND_Y_OFFSET = 110

    def __init__(self, neuron, ez_printer):
        # External API - preserve these attributes
        self.neuron             = neuron
        self.font               = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)
        diamond_center_x        = self.DIAMOND_OFFSET_FROM_OUTPUT + self.neuron.model.neurons[-1][-1].location_left + self.neuron.model.neurons[-1][-1].location_width
        self.diamond_center     = (diamond_center_x, self.DIAMOND_Y_OFFSET)

        self.arrow_targets      = [(diamond_center_x, self.DIAMOND_Y_OFFSET - self.DIAMOND_RADIUS * 2),
                                   (diamond_center_x + self.DIAMOND_RADIUS * 2, self.DIAMOND_Y_OFFSET ),
                                   (diamond_center_x + self.DIAMOND_RADIUS * 2 , self.DIAMOND_Y_OFFSET - self.DIAMOND_RADIUS)
                                   ]

    def render(self):

        # Get current iteration data
        rs = Const.dm.get_sample_data (self.neuron.run_id)
        is_correct = rs.get("is_true")

        # Get labels
        alpha_text = self.neuron.TRI.BD.label_min  # NO/False
        alpha_value = self.neuron.TRI.BD.target_min
        alpha_display = f"{round(alpha_value)} - {alpha_text}"

        beta_text = self.neuron.TRI.BD.label_max  # YES/True
        beta_value = self.neuron.TRI.BD.target_max
        beta_display = f"{round(beta_value)} - {beta_text}"

        # Determine what was ACTUALLY predicted using the classification flags       # These flags already capture the threshold logic correctly
        predicted_positive = (rs.get("is_true_positive") or rs.get("is_false_positive"))
        predicted_negative = (rs.get("is_true_negative") or rs.get("is_false_negative"))

        # Color YES branch (beta, positive, 1, True)
        if predicted_positive:  # Model chose YES
            if is_correct:  beta_color = get_darker_color(Const.COLOR_GREEN_FOREST)  # Correct
            else:           beta_color = get_darker_color(Const.COLOR_FOR_ACT_NEGATIVE)  # Wrong
        else:               beta_color = (100, 100, 120)  # Inactive gray # Model did NOT choose YES

        # Color NO branch (alpha, negative, 0, False)
        if predicted_negative:  # Model chose NO
            if is_correct: alpha_color = get_darker_color(Const.COLOR_GREEN_FOREST)  # Correct
            else:          alpha_color = get_darker_color(Const.COLOR_FOR_ACT_NEGATIVE)  # Wrong
        else:              alpha_color = (100, 100, 120)  # Inactive gray # Model did NOT choose NO

        # Draw everything
        threshold = self.neuron.TRI.BD.threshold
        self.draw_diamond_shape(self.diamond_center, self.DIAMOND_RADIUS, Const.COLOR_BLUE, self.font,f"> {threshold}", Const.COLOR_BLUE_PURE, 2)
        self.draw_rounded_text_box(self.arrow_targets[0], alpha_display,  Const.COLOR_WHITE, alpha_color, self.font)
        self.draw_rounded_text_box(self.arrow_targets[2], beta_display,   Const.COLOR_WHITE, beta_color, self.font)
    def draw_rounded_text_box(self, location, text, text_color, bg_color, font, x_align='center', padding=5, corner_radius=5):

        x, y = location
        text_width, text_height = font.size(text)                   # Measure text
        rect_width = text_width + (padding * 2)                     # Calculate rectangle dimensions
        rect_height = text_height + (padding * 2)                   # Calculate rectangle dimensions
        if x_align == 'center': rect_x = x - (rect_width // 2)      # Position rectangle in LOCAL coordinates
        else: rect_x = x  # 'left'
        rect_y = y - rect_height  # y is BOTTOM

        global_top_left = self.to_global_coords(rect_x, rect_y)     # Convert to GLOBAL coordinates
        global_rect = pygame.Rect(global_top_left[0], global_top_left[1], rect_width, rect_height)
        global_center = self.to_global_coords(rect_x + rect_width // 2, rect_y + rect_height // 2)

        # Schedule filled rounded rectangle (background)
        Const.dm.schedule_draw(pygame.draw.rect, Const.SCREEN, bg_color, global_rect, border_radius=corner_radius)

        # Schedule border (2pt black)
        Const.dm.schedule_draw(pygame.draw.rect, Const.SCREEN, (0, 0, 0), global_rect, width=2, border_radius=corner_radius)

        # Schedule text centered in rectangle
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=global_center)
        Const.dm.schedule_draw(Const.SCREEN.blit, text_surface, text_rect)

    def draw_diamond_shape(self,  center, radius, color, font, text=None, border_color=None, border_thickness=1):
        """Draw a diamond shape with optional text and border"""
        cx, cy = center

        # Calculate points in local coordinates
        local_points = [
            (cx, cy - radius),  # top
            (cx + radius, cy),  # right
            (cx, cy + radius),  # bottom
            (cx - radius, cy),  # left
        ]

        # Store points for external reference
        self.diamond_top = local_points[0]
        self.diamond_right = local_points[1]
        self.diamond_left = local_points[3]

        x, y = self.diamond_left
        self.point_prediction = (x + 20, y)
        global_points = [self.to_global_coords(x, y) for x, y in local_points] # Convert to global coordinates for drawing

        # Schedule filled diamond
        Const.dm.schedule_draw(pygame.draw.polygon, Const.SCREEN, color, global_points)

        # Schedule optional border
        if border_color and border_thickness > 0:
            Const.dm.schedule_draw(pygame.draw.polygon, Const.SCREEN, border_color, global_points, border_thickness)

        # Schedule optional centered text
        if text:
            text_surface = font.render(str(text), True, Const.COLOR_WHITE)
            text_rect = text_surface.get_rect(center=self.to_global_coords(cx, cy))
            Const.dm.schedule_draw(Const.SCREEN.blit, text_surface, text_rect)

    def to_global_coords(self, x, y):
        """Convert local neuron coordinates to global screen coordinates"""
        return (x + self.neuron.model.left, y + self.neuron.model.top)
