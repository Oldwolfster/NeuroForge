import pygame
from src.NNA.utils.general_text import smart_format


def draw_gradient_rect(surface, rect, color1, color2, border_radius=0):
    """
    Draws a gradient rectangle with optional rounded corners.

    :param surface: Pygame surface to draw on
    :param rect: pygame.Rect defining position and size
    :param color1: RGB color for top of gradient
    :param color2: RGB color for bottom of gradient
    :param border_radius: Radius for rounded corners (0 = sharp corners)
    """
    safe_height = min(rect.height, 1500)

    if border_radius > 0:
        # Create temporary surface for gradient with alpha channel
        temp_surface = pygame.Surface((rect.width, safe_height), pygame.SRCALPHA)

        # Draw gradient on temp surface
        for i in range(safe_height):
            ratio = i / safe_height
            blended_color = [
                int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
            ]
            pygame.draw.line(temp_surface, blended_color, (0, i), (rect.width, i))

        # Create mask with rounded corners
        mask_surface = pygame.Surface((rect.width, safe_height), pygame.SRCALPHA)
        pygame.draw.rect(mask_surface, (255, 255, 255, 255),
                         mask_surface.get_rect(), border_radius=border_radius)

        # Apply mask to gradient
        temp_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

        # Blit to main surface
        surface.blit(temp_surface, (rect.x, rect.y))
    else:
        # Original gradient drawing for sharp corners
        for i in range(safe_height):
            ratio = i / safe_height
            blended_color = [
                int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
            ]
            pygame.draw.line(surface, blended_color, (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))

def check_label_collision(new_label_rect, existing_labels_rects):
    """
    Checks if the new label's rect collides with any of the existing label rectangles.

    :param new_label_rect: A pygame.Rect representing the new label's boundaries.
    :param existing_labels_rects: A list of pygame.Rect objects for already placed labels.
    :return: True if there is a collision with any existing label, False otherwise.
    """
    for rect in existing_labels_rects:
        if new_label_rect.colliderect(rect):
            return True
    return False

def draw_rect_with_border(screen, rect, color, border_width, border_color=(0,0,0)):
    """
    Draws a rectangle with a border on the given Pygame surface.

    Parameters:
        screen (pygame.Surface): The surface to draw on.
        rect (pygame.Rect): The rectangle defining the position and size.
        color (tuple): The RGB color of the inner rectangle.
        border_color (tuple): The RGB color of the border.
        border_width (int): The thickness of the border.
    """
    # Draw the outer rectangle (border)
    pygame.draw.rect(screen, border_color, rect)

    # Calculate the dimensions of the inner rectangle
    inner_rect = rect.inflate(-2*border_width, -2*border_width)

    # Draw the inner rectangle
    #pygame.draw.rect(screen, color, inner_rect)
    draw_gradient_rect(screen, inner_rect, color, get_darker_color(color))

def get_text_rect(text: str, font_size: int):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, (0,0,0))
    return text_surface.get_rect()

def draw_text_with_background(screen, value_to_print, x, y, font_size, text_color=(255, 255, 255), bg_color=(0, 0, 0), right_align=False, border_color=None):
    """
    Draws text with a background rectangle for better visibility.

    :param right_align: If True, the text is right-aligned to x; otherwise, x is the left edge.
    :param border_color: If True, adds a black border
    """
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(smart_format(value_to_print), True, text_color)
    text_rect = text_surface.get_rect()

    # Original logic if right_align is False
    if not right_align:
        text_rect.topleft = (x, y)
    else:
        # If right_align is True, place the text so its right edge is at x
        text_rect.topright = (x, y)

    if not border_color is None:
        pygame.draw.rect(screen, border_color, text_rect.inflate(9, 7))  # Slight padding around text
        screen.blit(text_surface, text_rect)

    # Draw background rectangle
    pygame.draw.rect(screen, bg_color, text_rect.inflate(6, 4))  # Slight padding around text
    screen.blit(text_surface, text_rect)
def get_darker_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Given a background RGB color, this function returns an RGB tuple for a darker color,


    Parameters:
        rgb (tuple[int, int, int]): A tuple representing the background color (R, G, B).

    Returns:
        tuple[int, int, int]: An RGB tuple darker color
    """
    r, g, b = rgb
    towards_color = 11

    return (min(r+ towards_color, 255) / 2,min(g+ towards_color, 255) / 2,min(b+ towards_color, 255) / 2,)


def get_contrasting_text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Given a background RGB color, this function returns an RGB tuple for either black or white text,
    whichever offers better readability.

    The brightness is computed using the formula:
        brightness = (R * 299 + G * 587 + B * 114) / 1000
    which is a standard formula for perceived brightness. If the brightness is greater than 128,
    the background is considered light and black text is returned; otherwise, white text is returned.

    Parameters:
        rgb (tuple[int, int, int]): A tuple representing the background color (R, G, B).

    Returns:
        tuple[int, int, int]: An RGB tuple for the text color (either (0, 0, 0) for black or (255, 255, 255) for white).
    """
    r, g, b = rgb
    # Calculate the perceived brightness of the background color.
    brightness = (r * 299 + g * 587 + b * 114) / 1000

    # Choose black text for light backgrounds and white text for dark backgrounds.
    if brightness > 128:
        return (0, 0, 0)  # Black text for lighter backgrounds.
    else:
        return (255, 255, 255)  # White text for darker backgrounds.


def draw_gradient_rect( surface, rect, color1, color2):
    #print(f"rect.height={rect.height}")
    safe_height = min(rect.height, 1500)  # Clamp height to prevent hanging if height explodes. 2E31 lines drawn
    for i in range(safe_height):
        ratio = i / safe_height
        blended_color = [
            int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
        ]
        pygame.draw.line(surface, blended_color, (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))

def draw_gradient_rect(surface, rect, color1, color2, border_radius=0):
    """
    Draws a gradient rectangle with optional rounded corners.

    :param surface: Pygame surface to draw on
    :param rect: pygame.Rect defining position and size
    :param color1: RGB color for top of gradient
    :param color2: RGB color for bottom of gradient
    :param border_radius: Radius for rounded corners (0 = sharp corners)
    """
    safe_height = min(rect.height, 1500)

    if border_radius > 0:
        # Create temporary surface for gradient with alpha channel
        temp_surface = pygame.Surface((rect.width, safe_height), pygame.SRCALPHA)

        # Draw gradient on temp surface
        for i in range(safe_height):
            ratio = i / safe_height
            blended_color = [
                int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
            ]
            pygame.draw.line(temp_surface, blended_color, (0, i), (rect.width, i))

        # Create mask with rounded corners
        mask_surface = pygame.Surface((rect.width, safe_height), pygame.SRCALPHA)
        pygame.draw.rect(mask_surface, (255, 255, 255, 255),
                         mask_surface.get_rect(), border_radius=border_radius)

        # Apply mask to gradient
        temp_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

        # Blit to main surface
        surface.blit(temp_surface, (rect.x, rect.y))
    else:
        # Original gradient drawing for sharp corners
        for i in range(safe_height):
            ratio = i / safe_height
            blended_color = [
                int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
            ]
            pygame.draw.line(surface, blended_color, (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))
