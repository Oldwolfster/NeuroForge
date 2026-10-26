# ButtonBase.py                                                     CSR -- complete file
from typing import Callable, Tuple
import pygame
from src.NeuroForge import Const


class Button_Base:
    """
    On-screen button with shadow, press feedback, percentage or auto sizing,
    up to three lines of text, click handling and hover tooltips.

    auto_size=True sizes to fit whatever lines exist (plus padding);
    otherwise width_pct and height_pct are required.
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
        main_color = Const.COLOR_BLUE,
        shadow_color = Const.COLOR_FOR_SHADOW,
        border_radius: int = 61,
        font_size: int = 32,
        auto_size: bool = False,
        padding: int = 10,
        text_line2: str = None,
        text_color = Const.COLOR_WHITE,
        text_line2_color = None,
        text_line2_offset: int = 0,
        surface_offset: Tuple[int, int] = (0, 0),
        tooltip_text: str = None,
        text_line3: str = None,
    ):
        self.surface                    = my_surface
        self.offset_x, self.offset_y    = surface_offset
        self.font                       = pygame.font.SysFont(None, font_size)
        self.padding                    = padding
        self.store_text     (text, text_line2, text_line3, text_color, text_line2_color, text_line2_offset)
        self.store_style    (main_color, shadow_color, shadow_offset, border_radius)
        self.store_behavior (on_click, on_hover, tooltip_text)
        self.place_me       (left_pct, top_pct, width_pct, height_pct, auto_size)

    # ---------------------------------------------------------------- construction

    def store_text(self, text, text_line2, text_line3, text_color, text_line2_color, text_line2_offset):
        self.text               = text
        self.text_line2         = text_line2
        self.text_line3         = text_line3
        self.text_color         = text_color
        self.text_line2_color   = text_line2_color if text_line2_color is not None else text_color
        self.text_line2_offset  = text_line2_offset

    def store_style(self, main_color, shadow_color, shadow_offset, border_radius):
        self.main_color     = main_color
        self.shadow_color   = shadow_color
        self.shadow_offset  = shadow_offset
        self.border_radius  = border_radius

    def store_behavior(self, on_click, on_hover, tooltip_text):
        self.on_click       = on_click
        self.on_hover       = on_hover
        self.tooltip_text   = tooltip_text
        self.is_pressed     = False

    def place_me(self, left_pct, top_pct, width_pct, height_pct, auto_size):
        pw, ph                  = self.surface.get_size()
        self.left               = int(pw * (left_pct / 100))
        self.top                = int(ph * (top_pct  / 100))
        self.width, self.height = self.measure_lines() if auto_size else self.measure_pct(pw, ph, width_pct, height_pct)
        self.button_rect        = pygame.Rect(self.left, self.top, self.width, self.height)

    def measure_pct(self, pw, ph, width_pct, height_pct) -> tuple:
        return int(pw * (width_pct / 100)), int(ph * (height_pct / 100))

    def measure_lines(self) -> tuple:
        """Widest line wins the width; every line adds to the height."""
        sizes = [self.font.size(text) for text in self.line_texts()]
        return (max(w for w, h in sizes) + self.padding * 2,
                sum(h for w, h in sizes) + self.padding * 2)

    # ---------------------------------------------------------------- the lines

    def line_specs(self) -> list:
        """Text and color travel together, so a skipped line cannot shift the colors."""
        pairs = ((self.text,       self.text_color),
                 (self.text_line2, self.text_line2_color),
                 (self.text_line3, self.text_color))
        return [(text, color) for text, color in pairs if text]

    def line_texts(self) -> list:
        return [text for text, color in self.line_specs()]

    # ---------------------------------------------------------------- drawing

    def draw_me(self):
        if not self.is_pressed: self.draw_shadow()
        draw_rect = self.button_rect.move(2, 2) if self.is_pressed else self.button_rect
        self.draw_body(draw_rect)
        self.draw_text(draw_rect)

    def draw_shadow(self):
        """Vertical always falls; horizontal follows the sign the caller passed."""
        rect = self.button_rect.move(self.shadow_offset, abs(self.shadow_offset))
        pygame.draw.rect(self.surface, self.shadow_color, rect, border_radius=self.border_radius)

    def draw_body(self, draw_rect):
        pygame.draw.rect(self.surface, self.main_color, draw_rect, border_radius=self.border_radius)

    def draw_text(self, draw_rect):
        specs  = self.line_specs()
        bottom = self.text_top(draw_rect, len(specs))
        for text, color in specs:
            surf   = self.font.render(text, True, color)
            rect   = surf.get_rect(center=(draw_rect.centerx, bottom + surf.get_height() / 2))
            self.surface.blit(surf, rect)
            bottom = rect.bottom + self.padding / 2 + self.text_line2_offset

    def text_top(self, draw_rect, count: int) -> float:
        """A lone line centers in the button; a stack starts at the top pad."""
        if count == 1:  return draw_rect.centery - self.font.get_height() / 2
        return draw_rect.top + self.padding

    # ---------------------------------------------------------------- tooltip

    def render_tooltip(self):
        if self.on_hover is not None:   return self.on_hover()
        if self.tooltip_text:           self.draw_simple_tooltip()

    def draw_simple_tooltip(self):
        font    = pygame.font.SysFont(None, 24)
        surf    = font.render(self.tooltip_text, True, Const.COLOR_BLACK)
        pad     = 6
        w, h    = surf.get_width() + pad * 2, surf.get_height() + pad * 2
        x, y    = self.tooltip_position(w, h)
        pygame.draw.rect(Const.SCREEN, Const.COLOR_FOR_POPUP, (x, y, w, h))
        pygame.draw.rect(Const.SCREEN, Const.COLOR_BLACK,     (x, y, w, h), 1)
        Const.SCREEN.blit(surf, (x + pad, y + pad))

    def tooltip_position(self, w: int, h: int) -> tuple:
        """Flips to the near side of the cursor rather than running off screen."""
        mx, my = pygame.mouse.get_pos()
        x = mx + 12 if mx + 12 + w <= Const.SCREEN_WIDTH  else mx - w - 4
        y = my + 12 if my + 12 + h <= Const.SCREEN_HEIGHT else my - h - 4
        return x, y

    # ---------------------------------------------------------------- events

    def process_an_event(self, event):
        if   event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:    self.press(event)
        elif event.type == pygame.MOUSEBUTTONUP   and event.button == 1:    self.release(event)

    def press(self, event):
        if self.hit_test(event.pos):    self.is_pressed = True

    def release(self, event):
        """Fires only when the release lands on the same button that took the press."""
        if not self.is_pressed: return
        self.is_pressed = False
        if self.hit_test(event.pos) and callable(self.on_click):    self.on_click()

    def hit_test(self, pos) -> bool:
        return self.button_rect.collidepoint((pos[0] - self.offset_x, pos[1] - self.offset_y))

    def update_me(self):
        pass

    def get_global_rect(self):
        return pygame.Rect(self.left + self.offset_x, self.top + self.offset_y, self.width, self.height)