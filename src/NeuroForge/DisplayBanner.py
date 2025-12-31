# DisplayBanner.py

import pygame


from src.NeuroForge.EZSurface import EZSurface

from src.NNA.utils.general_text import beautify_text
from src.NeuroForge import Const
from src.NNA.engine.TrainingData import TrainingData


class DisplayBanner(EZSurface):
    def __init__(self, training_data: TrainingData, max_epoch: int, max_sample: int,
                 width_pct=98, height_pct=4.369, left_pct=1, top_pct=0):
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)
        self.child_name = "Top Banner"
        self.training_data = training_data
        self.max_epoch = max_epoch
        self.max_sample = max_sample
        self.banner_text_left = "Loading..."
        self.banner_text_right = ""
        self.banner_text_center = "Neuro Forge"
        self.font = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 69)

    def update_me(self):
        epoch = Const.vcr.CUR_EPOCH
        sample = Const.vcr.CUR_SAMPLE
        self.banner_text_left = f"Epoch: {epoch}/{self.max_epoch} Sample: {sample}/{self.max_sample}"
        self.banner_text_right = f"{beautify_text(self.training_data.arena_name)}: {self.training_data.problem_type}"

    def render(self):
        self.clear()
        margin = 8

        # Left text
        label_left = self.font.render(self.banner_text_left, True, Const.COLOR_WHITE)

        # Right text
        label_right = self.font.render(self.banner_text_right, True, Const.COLOR_WHITE)
        right_x = self.width - label_right.get_width() - margin

        # Center title with glow
        glow, text = self.render_glowing_text(self.banner_text_center, self.font_large)
        center_x = (self.width - text.get_width()) // 2
        center_y = (self.height - text.get_height()) // 2 + 3

        # Border
        pygame.draw.rect(self.surface, Const.COLOR_BLACK, (0, 0, self.width, self.height), 4)

        # Blit all
        self.surface.blit(label_left, (margin, margin))
        self.surface.blit(label_right, (right_x, margin))
        self.surface.blit(glow, (center_x - 5, center_y - 5))
        self.surface.blit(text, (center_x, center_y))

    def render_glowing_text(self, text, font, glow_strength=5):
        base = font.render(text, True, Const.COLOR_MOLTEN)
        glow = font.render(text, True, Const.COLOR_MOLTEN_GLOW)
        for _ in range(glow_strength):
            glow = pygame.transform.smoothscale(glow, (glow.get_width() + 1, glow.get_height() + 1))
        return glow, base