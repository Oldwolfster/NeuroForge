# TextBox.py
from typing import Callable
import pygame
from src.NeuroForge import Const


class TextBox:
    """
    Text input box with shadow, percentage-based sizing, matching Button_Base style.
    """
    def __init__(
        self,
        my_surface: pygame.Surface,
        left_pct: float,
        top_pct: float,
        width_pct: float,
        height_pct: float,
        on_submit: Callable[[str], None] = None,
        placeholder: str = "",
        initial_value: str = "",
        shadow_offset_x: int = -3,
        shadow_offset_y: int = 3,
        bg_color=Const.COLOR_FOR_TXT_BG,
        border_color=Const.COLOR_FOR_TXT_BORDER,
        text_color=Const.COLOR_FOR_TXT_TXT,
        placeholder_color=Const.COLOR_FOR_TXT_PH,
            prompt: str = "",
        font_size: int = 20,
        surface_offset: tuple = (0, 0),
        numeric_only: bool = False,
    ):
        self.surface        = my_surface
        self.offset_x       = surface_offset[0]
        self.offset_y       = surface_offset[1]
        pw, ph              = self.surface.get_size()

        self.left           = int(pw * (left_pct / 100))
        self.top            = int(ph * (top_pct / 100))
        self.width          = int(pw * (width_pct / 100))
        self.height         = int(ph * (height_pct / 100))

        self.on_submit      = on_submit
        self.placeholder    = placeholder
        self.prompt         = prompt
        self.text           = initial_value
        self.numeric_only   = numeric_only

        self.shadow_x       = shadow_offset_x
        self.shadow_y       = shadow_offset_y
        self.bg_color       = bg_color
        self.border_color   = border_color
        self.text_color     = text_color
        self.placeholder_color = placeholder_color

        self.font           = pygame.font.SysFont(None, font_size)
        self.rect           = pygame.Rect(self.left, self.top, self.width, self.height)
        self.is_focused     = False
        self.cursor_visible = True
        self.cursor_timer   = 0

    def update_me(self):
        pass

    def draw_me(self):
        # Shadow

        shadow_rect = self.rect.move((self.shadow_x),(self.shadow_y))
        pygame.draw.rect(self.surface, Const.COLOR_FOR_SHADOW, shadow_rect, border_radius=4)

        # Background
        pygame.draw.rect(self.surface, self.bg_color, self.rect, border_radius=4)

        # Border (thicker if focused)
        border_width = 3 if self.is_focused else 2
        pygame.draw.rect(self.surface, self.border_color, self.rect, border_width, border_radius=4)

        # Text or placeholder
        if self.text:
            display_text    = self.text
            color           = self.text_color
        elif self.is_focused and self.prompt:
            display_text    = self.prompt
            color           = self.placeholder_color
        else:
            display_text    = self.placeholder
            color           = self.placeholder_color
        color               = self.text_color if self.text else self.placeholder_color
        text_surf           = self.font.render(display_text, True, color)
        text_rect           = text_surf.get_rect(midleft=(self.rect.left + 8, self.rect.centery))
        self.surface.blit(text_surf, text_rect)

        # Cursor
        if self.is_focused:
            self.cursor_timer += 1
            if self.cursor_timer % 60 < 30:
                cursor_x = text_rect.right + 2 if self.text else self.rect.left + 10
                cursor_y1 = self.rect.centery - 10
                cursor_y2 = self.rect.centery + 10
                pygame.draw.line(self.surface, self.text_color, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)

    def process_an_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            lx = event.pos[0] - self.offset_x
            ly = event.pos[1] - self.offset_y
            self.is_focused = self.rect.collidepoint(lx, ly)
            if self.is_focused:
                self.cursor_timer = 0

        if event.type == pygame.KEYDOWN and self.is_focused:
            if event.key == pygame.K_RETURN:
                self.submit()
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.is_focused = False
            else:
                char = event.unicode
                if char and self.is_valid_char(char):
                    self.text += char

    def is_valid_char(self, char):
        if not char.isprintable():
            return False
        if self.numeric_only:
            return char.isdigit()
        return True

    def submit(self):
        if self.on_submit and self.text:
            self.on_submit(self.text)
        self.text       = ""  # ADD THIS
        self.is_focused = False

    def get_global_rect(self):
        return pygame.Rect(self.left + self.offset_x,
                           self.top  + self.offset_y,
                           self.width,
                           self.height)

    def clear(self):
        self.text = ""