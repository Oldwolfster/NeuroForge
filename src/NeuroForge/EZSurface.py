from abc import ABC, abstractmethod
import pygame
from src.NeuroForge import Const


class EZSurface(ABC):
    """
    Abstract base class for Display_Manager components.
    Provides:
    1) Standard methods for updating and rendering
    2) Independent surface with percentage-based positioning (resolution independence)
    """
    __slots__ = (
        "screen_width", "screen_height",
        "left_pct", "width_pct", "top_pct", "height_pct",
        "width", "height", "left", "top",
        "surface", "bg_color"
    )

    def __init__(self, width_pct=100, height_pct=100, left_pct=0, top_pct=0,
                 bg_color=Const.COLOR_WHITE, transparent=False,
                 pixel_adjust_width=0, pixel_adjust_height=0,
                 pixel_adjust_left=0, pixel_adjust_top=0):
        self.screen_width  = Const.SCREEN_WIDTH
        self.screen_height = Const.SCREEN_HEIGHT
        self.left_pct      = left_pct
        self.width_pct     = width_pct
        self.top_pct       = top_pct
        self.height_pct    = height_pct

        # Calculate dimensions from percentages
        self.width  = int(self.screen_width  * (width_pct  / 100)) + pixel_adjust_width
        self.height = int(self.screen_height * (height_pct / 100)) + pixel_adjust_height
        self.left   = int(self.screen_width  * (left_pct   / 100)) + pixel_adjust_left
        self.top    = int(self.screen_height * (top_pct    / 100)) + pixel_adjust_top

        # Create surface
        flags = pygame.SRCALPHA if transparent else 0
        self.surface = pygame.Surface((self.width, self.height), flags)
        self.bg_color = bg_color
        self.surface.fill(self.bg_color)

    def get_global_rect(self):
        return pygame.Rect(self.left, self.top, self.width, self.height)

    @abstractmethod
    def render(self):
        """Render custom content - implemented by child classes."""
        pass

    def update_me(self):
        """Update state - override in child classes if needed."""
        pass

    def draw_me(self):
        """Clear, render, and blit to main screen."""
        self.clear()
        self.render()
        Const.SCREEN.blit(self.surface, (self.left, self.top))

    def clear(self):
        """Clear the surface."""
        if self.surface.get_flags() & pygame.SRCALPHA:
            self.surface.fill(Const.COLOR_TRANSPARENT)
        else:
            self.surface.fill(self.bg_color)

    def resize(self, new_width_pct=None, new_height_pct=None):
        """Dynamically resize while maintaining position."""
        if new_width_pct:
            self.width_pct = new_width_pct
            self.width = int(self.screen_width * (new_width_pct / 100))
        if new_height_pct:
            self.height_pct = new_height_pct
            self.height = int(self.screen_height * (new_height_pct / 100))

        flags = pygame.SRCALPHA if self.surface.get_flags() & pygame.SRCALPHA else 0
        self.surface = pygame.Surface((self.width, self.height), flags)
        self.clear()