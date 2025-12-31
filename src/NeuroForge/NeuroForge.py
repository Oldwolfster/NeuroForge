# NeuroForge.py

import pygame
from src.NeuroForge import Const
from src.NeuroForge.ButtonMenu import ButtonMenu
from src.NeuroForge.Display_Manager import Display_Manager
from src.NeuroForge.VCR import VCR
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from typing import List


def initialize_display(fullscreen):
    """Initialize display dimensions. Call after pygame.init()."""
    if fullscreen:
        display_info = pygame.display.Info()
        Const.SCREEN_WIDTH = display_info.current_w
        Const.SCREEN_HEIGHT = display_info.current_h
        Const.SCREEN = pygame.display.set_mode((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT), pygame.FULLSCREEN)
    else:
        Const.SCREEN_WIDTH = 1900
        Const.SCREEN_HEIGHT = 900
        Const.SCREEN = pygame.display.set_mode((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))

    # Compute derived values
    Const.MODEL_AREA_PIXELS_LEFT = Const.SCREEN_WIDTH * Const.MODEL_AREA_PERCENT_LEFT
    Const.MODEL_AREA_PIXELS_TOP = Const.SCREEN_HEIGHT * Const.MODEL_AREA_PERCENT_TOP
    Const.MODEL_AREA_PIXELS_WIDTH = Const.SCREEN_WIDTH * Const.MODEL_AREA_PERCENT_WIDTH
    Const.MODEL_AREA_PIXELS_HEIGHT = Const.SCREEN_HEIGHT * Const.MODEL_AREA_PERCENT_HEIGHT


def quit_neuroforge():
    """Clean shutdown - can be called from anywhere."""
    Const.IS_RUNNING = False


def NeuroForge(TRIs: List[TrainingRunInfo], fullscreen=False):
    """Initialize NeuroForge and run the visualization loop."""

    if len(TRIs) > 2:
        print(f"⚠️ NeuroForge supports max 2 models. Showing first 2 of {len(TRIs)}.")
        TRIs = TRIs[:2]

    pygame.init()
    initialize_display(fullscreen)
    pygame.display.set_caption("NeuroForge")

    Const.vcr = VCR()
    Const.dm = Display_Manager(TRIs)

    menu_button = ButtonMenu()
    clock = pygame.time.Clock()

    while Const.IS_RUNNING:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:  quit_neuroforge()
            if event.type == pygame.QUIT:   quit_neuroforge()
            Const.dm.process_events(event)
            menu_button.handle_event(event)
        Const.dm.update()
        Const.SCREEN.fill(Const.COLOR_FOR_BACKGROUND)
        Const.dm.render()
        menu_button.draw()
        Const.dm.render_popup()
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()