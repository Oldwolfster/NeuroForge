# EZFormLEFT.py

from typing import Dict
import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZSurface import EZSurface


class EZForm(EZSurface):
    """Form panel with banner, shadow, and dynamic key-value fields."""

    def __init__(self, fields: Dict[str, str], width_pct: int, height_pct: int,
                 left_pct: int, top_pct: int, banner_text="Form",
                 banner_color=Const.COLOR_BLUE, bg_color=Const.COLOR_FOR_BACKGROUND,
                 font_color=Const.COLOR_BLACK, shadow_offset=6, same_line=False, hover_popup=None):

        self.fields = fields
        self.banner_text = banner_text
        self.banner_color = banner_color
        self.font_color = font_color
        self.same_line = same_line
        self.shadow_offset = shadow_offset
        self.hover_popup = hover_popup

        # Fonts
        self.banner_font = pygame.font.Font(None, 36)
        self.field_font = pygame.font.Font(None, 24)

        # For arrow positioning
        self.label_y_positions = []
        self.need_label_coord = True

        # Calculate form dimensions
        self.form_width = int(Const.SCREEN_WIDTH * (width_pct / 100))
        self.form_height = int(Const.SCREEN_HEIGHT * (height_pct / 100))

        # Parent surface is slightly larger to accommodate shadow
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color, False,
                         shadow_offset, shadow_offset, 0, 0)

        # Pre-calculate rects
        self.shadow_rect = pygame.Rect(0, shadow_offset, self.form_width, self.form_height)
        self.form_rect = pygame.Rect(shadow_offset, 0, self.form_width, self.form_height)
        self.banner_height = self.banner_font.get_height() + 8
        self.banner_rect = pygame.Rect(shadow_offset, 0, self.form_width, self.banner_height)

        self.render()  # Capture label positions

    def render(self):
        self.clear()
        self.draw_shadow()
        self.draw_background()
        self.draw_banner()
        self.draw_border()
        self.draw_fields()

    def draw_shadow(self):
        pygame.draw.rect(self.surface, Const.COLOR_FOR_SHADOW, self.shadow_rect,
                         border_radius=self.shadow_offset)

    def draw_background(self):
        pygame.draw.rect(self.surface, Const.COLOR_FOR_BACKGROUND, self.form_rect,
                         border_radius=4)

    def draw_banner(self):
        pygame.draw.rect(self.surface, self.banner_color, self.banner_rect, border_radius=5)
        text_surface = self.banner_font.render(self.banner_text, True, Const.COLOR_WHITE)
        text_rect = text_surface.get_rect(center=(self.form_rect.centerx, self.banner_height // 2))
        self.surface.blit(text_surface, text_rect)

    def draw_border(self):
        new_rect = self.form_rect.copy()
        new_rect.top += 2
        new_rect.height -= 2
        pygame.draw.rect(self.surface, self.banner_color, new_rect, 3, border_radius=4)

    def draw_fields(self):
        if not self.fields:
            return

        spacing = 8
        field_start_y = self.banner_height + spacing - (5 if self.same_line else 0)
        field_spacing = (self.height - field_start_y) // len(self.fields)
        box_margin = int(self.form_width * 0.05)

        for i, (label, value) in enumerate(self.fields.items()):
            y_pos = field_start_y + (i * field_spacing) + 20
            box_rect = pygame.Rect(box_margin + self.shadow_offset, y_pos - 15,
                                   self.form_width - (2 * box_margin), 30)

            self.track_label_position(y_pos)
            self.draw_field_box(box_rect)
            self.draw_field_content(box_rect, label, value)

        if self.label_y_positions:
            self.need_label_coord = False

    def track_label_position(self, y_pos):
        """Track label positions for arrow drawing."""
        if self.need_label_coord:
            global_x = self.left + self.width + 16
            global_y = self.top + y_pos
            self.label_y_positions.append((global_x, global_y))

    def draw_field_box(self, box_rect):
        pygame.draw.rect(self.surface, Const.COLOR_WHITE, box_rect)
        pygame.draw.rect(self.surface, self.banner_color, box_rect, 2)

    def draw_field_content(self, box_rect, label, value):
        if self.same_line:
            self.draw_field_same_line(box_rect, label, value)
        else:
            self.draw_field_stacked(box_rect, label, value)

    def draw_field_same_line(self, box_rect, label, value):
        """Label left, value right, same line."""
        value_color = self.get_value_color(value)

        label_surface = self.field_font.render(label, True, self.font_color)
        label_rect = label_surface.get_rect(left=box_rect.left + 8, centery=box_rect.centery)
        self.surface.blit(label_surface, label_rect)

        value_surface = self.field_font.render(str(value), True, value_color)
        value_rect = value_surface.get_rect(right=box_rect.right - 8, centery=box_rect.centery)
        self.surface.blit(value_surface, value_rect)

    def draw_field_stacked(self, box_rect, label, value):
        """Label above, value centered below."""
        label_surface = self.field_font.render(label, True, self.font_color)
        label_rect = label_surface.get_rect(left=8 + self.shadow_offset,
                                            centery=box_rect.top - 5)
        self.surface.blit(label_surface, label_rect)

        value_surface = self.field_font.render(str(value), True, self.font_color)
        value_rect = value_surface.get_rect(center=box_rect.center)
        self.surface.blit(value_surface, value_rect)

    def get_value_color(self, value):
        """Gray for zero values, normal otherwise."""
        try:
            return Const.COLOR_GRAY if float(value) == 0 else self.font_color
        except (ValueError, TypeError):
            return Const.COLOR_GRAY if str(value) == "0" else self.font_color

    def set_colors(self, correct: bool):
        """Update banner color based on correctness."""
        if correct:
            self.banner_color = Const.COLOR_BANNER_CORRECT
        else:
            self.banner_color = Const.COLOR_BANNER_WRONG

    def render_tooltip(self):
        if self.hover_popup:
            self.hover_popup.show_me()