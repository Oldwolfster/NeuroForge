import pygame
from src.NeuroForge import Const

class ButtonMenu:
    """Handles the rendering and interaction of the Menu button."""

    def __init__(self):
        """Initialize the menu button position and properties."""
        self.width = 140
        self.height = 40
        self.top = 40
        self.left = Const.SCREEN_WIDTH - 26 - self.width  # Positioning on the right side
        self.button_rect = pygame.Rect(self.left, self.top, self.width, self.height)
        self.main_button_color = Const.COLOR_BLUE
        self.shadow_color = Const.COLOR_FOR_SHADOW  # Darker blue for depth
        self.border_radius = 10  # Rounded button corners

    def draw(self):
        """Draw the menu button on the screen."""
        shadow_offset = 5
        shadow_rect = self.button_rect.move(shadow_offset, abs(shadow_offset))

        pygame.draw.rect(Const.SCREEN, self.shadow_color, shadow_rect, border_radius=abs(shadow_offset) * 2)
        pygame.draw.rect(Const.SCREEN, self.main_button_color, self.button_rect, border_radius=abs(shadow_offset) * 2)

        font = pygame.font.SysFont(None, 32)
        text_surface = font.render("Menu", True, Const.COLOR_WHITE)
        text_rect = text_surface.get_rect(center=self.button_rect.center)
        Const.SCREEN.blit(text_surface, text_rect)

    def handle_event(self, event):
        """Check for mouse clicks and activate the menu if clicked."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.button_rect.collidepoint(event.pos):
            Const.MENU_ACTIVE = True


class ButtonInfo:
    """Handles the rendering and interaction of the Menu button."""

    def __init__(self):
        """Initialize the menu button position and properties."""
        self.width = 140
        self.height = 40
        self.top = 40
        self.left =  20   # Positioning on the left side
        self.button_rect = pygame.Rect(self.left, self.top, self.width, self.height)
        self.main_button_color = Const.COLOR_BLUE
        self.shadow_color = Const.COLOR_FOR_SHADOW  # Darker blue for depth
        self.border_radius = 10  # Rounded button corners

    def draw(self):
        """Draw the menu button on the screen."""
        shadow_offset = -5
        shadow_rect = self.button_rect.move(shadow_offset, abs(shadow_offset))

        pygame.draw.rect(Const.SCREEN, self.shadow_color, shadow_rect, border_radius=abs(shadow_offset) * 2)
        pygame.draw.rect(Const.SCREEN, self.main_button_color, self.button_rect, border_radius=abs(shadow_offset) * 2)

        font = pygame.font.SysFont(None, 32)
        text_surface = font.render("Sample", True, Const.COLOR_WHITE)
        text_rect = text_surface.get_rect(center=self.button_rect.center)
        Const.SCREEN.blit(text_surface, text_rect)

    def handle_event(self, event):
        """Check for mouse clicks and activate the menu if clicked."""
        if event.type == pygame.MOUSEBUTTONDOWN and self.button_rect.collidepoint(event.pos):
            Const.MENU_ACTIVE = True
