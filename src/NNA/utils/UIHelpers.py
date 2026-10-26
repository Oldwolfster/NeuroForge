import pygame
from src.NeuroForge import Const


def render_glowing_text(text, font, glow_strength=5):
    """
    Render glowing text using the molten color scheme.
    Returns: (glow_surface, text_surface) tuple
    """
    base = font.render(text, True, Const.COLOR_MOLTEN)
    glow = font.render(text, True, Const.COLOR_MOLTEN_GLOW)
    for _ in range(glow_strength):
        glow = pygame.transform.smoothscale(glow, (glow.get_width() + 1, glow.get_height() + 1))
    return glow, base

def render_glowing_text2(text, font, glow_strength=5):
    """
    Render glowing text using the molten color scheme.
    Returns: (glow_surface, text_surface) tuple
    """
    base = font.render(text, True, Const.COLOR_MOLTEN)

    glow_raw = font.render(text, True, Const.COLOR_MOLTEN_GLOW)

    # Give the glow some transparent breathing room so scaling doesn't "clip" at edges
    pad = glow_strength + 2
    glow = pygame.Surface(
        (glow_raw.get_width() + pad * 2, glow_raw.get_height() + pad * 2),
        pygame.SRCALPHA
    )
    glow.blit(glow_raw, (pad, pad))

    for _ in range(glow_strength):
        glow = pygame.transform.smoothscale(
            glow,
            (glow.get_width() + 1, glow.get_height() + 1)
        )

    return glow, base


def render_glowing_text3( text, font, glow_strength=3):
    """
    Render glowing text using the molten color scheme.
    Returns: (glow_surface, text_surface) tuple
    """
    base = font.render(text, True, Const.COLOR_MOLTEN)

    glow_raw = font.render(text, True, Const.COLOR_MOLTEN_GLOW)

    # Give the glow some transparent breathing room so scaling doesn't "clip" at edges
    pad = glow_strength + 2
    glow = pygame.Surface(
        (glow_raw.get_width() + pad * 2, glow_raw.get_height() + pad * 2),
        pygame.SRCALPHA
    )
    glow.blit(glow_raw, (pad, pad))

    for _ in range(glow_strength):
        glow = pygame.transform.smoothscale(
            glow,
            (glow.get_width() + 1, glow.get_height() + 1)
        )

    return glow, base

def render_glowing_text5( text, font, glow_strength=5, pad=3):
    """
    Render glowing text using the molten color scheme.
    Returns: (glow_surface, text_surface) tuple
    """
    base = font.render(text, True, Const.COLOR_MOLTEN)
    glow = font.render(text, True, Const.COLOR_MOLTEN_GLOW)

    for _ in range(glow_strength):
        glow = pygame.transform.smoothscale(
            glow, (glow.get_width() + 1, glow.get_height() + 1)
        )

    # Add buffer *after* the effect is created (preserves the look)
    if pad:
        padded = pygame.Surface((glow.get_width() + pad * 2, glow.get_height() + pad * 2), pygame.SRCALPHA)
        padded.blit(glow, (pad, pad))
        glow = padded

    return glow, base