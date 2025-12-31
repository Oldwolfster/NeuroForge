from typing import Callable, Tuple
import pygame
from src.NeuroForge import Const

class Button_Base:
    """
    Base class for on-screen buttons with shadow, percentage-based sizing, two-line text support, and click handling.

    Use `auto_size=True` to size width/height to fit text lines (with padding), otherwise `width_pct` and `height_pct` are required.
    """
    def __init__(
        self,
        my_surface: pygame.Surface,
        text: str,
        left_pct: float,
        top_pct: float,
        height_pct: float = 0,
        width_pct: float = 0,
        on_click: Callable = None,
        on_hover: Callable = None,
        shadow_offset: int = 5,
        main_color=Const.COLOR_BLUE,
        shadow_color=Const.COLOR_FOR_SHADOW,
        border_radius: int = 3,
        font_size: int = 32,
        auto_size: bool = False,
        padding: int = 10,
        text_line2: str = None,
        text_color = Const.COLOR_WHITE,      # NEW: color for first line (default white)
        text_line2_color = None,             # NEW: color for second line (defaults to same as line 1)
        text_line2_offset: int = 0,
        surface_offset: Tuple[int, int] = (0, 0),

    ):
        # parent surface for relative sizing
        self.surface = my_surface
        # store surface offset on the main screen
        self.offset_x, self.offset_y = surface_offset
        pw, ph = self.surface.get_size()

        # store text and font
        self.text = text
        self.text_line2 = text_line2
        self.font = pygame.font.SysFont(None, font_size)
        self.text_line2_offset = text_line2_offset

        # determine size: auto or percentage-based
        if auto_size:
            # measure first line
            w1, h1 = self.font.size(self.text)
            # measure second line if exists
            if self.text_line2:
                w2, h2 = self.font.size(self.text_line2)
            else:
                w2, h2 = (0, 0)
            # width is max of both plus padding
            self.width = max(w1, w2) + padding * 2
            # height is sum of line heights plus padding
            self.height = h1 + h2 + padding * 2
        else:
            self.width = int(pw * (width_pct / 100))
            self.height = int(ph * (height_pct / 100))

        # position offsets within the surface
        self.left = int(pw * (left_pct / 100))
        self.top = int(ph * (top_pct / 100))

        # final button rectangle in surface-local coords
        self.button_rect = pygame.Rect(self.left, self.top, self.width, self.height)

        # styling and behavior
        self.text_color = text_color
        # If line 2 color not specified, use same as line 1
        self.text_line2_color = text_line2_color if text_line2_color is not None else text_color
        self.on_click = on_click
        self.on_hover = on_hover
        self.shadow_offset = shadow_offset
        self.main_color = main_color
        self.shadow_color = shadow_color
        self.border_radius = border_radius
        self.padding = padding

    def update_me(self):
        pass

    def render_tooltip(self):
        if self.on_hover is not None:
            self.on_hover()


    def draw_me(self):
        # draw shadow
        #print(f"I am a button - my text is {self.text}")
        shadow_rect = self.button_rect.move(self.shadow_offset, abs(self.shadow_offset))
        pygame.draw.rect(
            self.surface,
            self.shadow_color,
            shadow_rect,
            border_radius=self.border_radius,
        )
        # draw main button
        pygame.draw.rect(
            self.surface,
            self.main_color,
            self.button_rect,
            border_radius=self.border_radius,
        )
        # draw first line
        text_surf1 = self.font.render(self.text, True, self.text_color)

        if self.text_line2:
            # Two lines: position first line near top
            rect1 = text_surf1.get_rect(
                center=(
                    self.button_rect.centerx,
                    self.button_rect.top + self.padding + text_surf1.get_height() / 2
                )
            )
            self.surface.blit(text_surf1, rect1)

            # Draw second line below first
            text_surf2 = self.font.render(self.text_line2, True, self.text_line2_color)
            rect2 = text_surf2.get_rect(
                center=(
                    self.button_rect.centerx,
                    rect1.bottom + text_surf2.get_height() / 2 + self.padding / 2 + self.text_line2_offset
                )
            )
            self.surface.blit(text_surf2, rect2)
        else:
            # Single line: center vertically in button
            rect1 = text_surf1.get_rect(center=self.button_rect.center)
            self.surface.blit(text_surf1, rect1)

    def process_an_event(self, event):
        """
        Handle mouse events, translating global coordinates into surface-local coordinates
        before checking for clicks on the button.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            gx, gy = event.pos
            # translate to this surface's local coordinates
            lx = gx - self.offset_x
            ly = gy - self.offset_y
            # if click falls within the button rect, fire callback
            if self.button_rect.collidepoint((lx, ly)):
                if callable(self.on_click):
                    self.on_click()

    def is_hovered(self, model_x, model_y, mouse_x, mouse_y):
        """Checks if mouse is in rectangle, translating for surface offset."""
        local_x = mouse_x - self.offset_x
        local_y = mouse_y - self.offset_y
        return self.button_rect.collidepoint((local_x, local_y))


