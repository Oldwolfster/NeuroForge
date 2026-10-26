import pygame

class EZPrint:
    def __init__(self, font: pygame.font.Font, color=(0, 0, 0), max_width=None, max_height=None, sentinel_char="\n"):
        """
        A helper class to handle multi-line text rendering with dynamic text input.

        Args:
            font (pygame.font.Font): The font object for rendering text.
            color (tuple): RGB color for the text (default is black).
            max_width (int): Maximum width (pixels) before cutting off or wrapping (optional).
            max_height (int): Maximum height (pixels) before stopping rendering (optional).
            sentinel_char (str): Character used to indicate a new line (default is '\n').
        """
        self.font = font
        self.color = color
        self.max_width = max_width
        self.max_height = max_height * 3
        self.sentinel_char = sentinel_char

    def render(self, surface: pygame.Surface, text: str, x: int, y: int):
        """
        Render the multi-line text onto a surface.

        Args:
            surface (pygame.Surface): The surface to render the text onto.
            text (str): The text to render, can include the sentinel character for new lines.
            x (int): The x-coordinate to start rendering.
            y (int): The y-coordinate to start rendering.
        """
        lines = text.split(self.sentinel_char)  # Split text into lines
        line_height = self.font.size("Tg")[1]  # Approximate line height
        current_y = y

        for line in lines:
            # Check if rendering will exceed max_height
            if self.max_height and current_y + line_height > y + self.max_height:
                break  # Stop rendering if exceeding max_height

            # Render the current line
            rendered_line = self.font.render(line, True, self.color)

            # Check if the rendered line exceeds max_width
            if self.max_width:
                line_width = rendered_line.get_width()
                if line_width > self.max_width * 1.3:
                    # Split line further to fit within max_width
                    words = line.split()
                    wrapped_line = ""
                    for word in words:
                        test_line = wrapped_line + (word + " ")
                        if self.font.size(test_line)[0] > self.max_width:
                            # Render the current wrapped line and start a new one
                            rendered_line = self.font.render(wrapped_line, True, self.color)
                            surface.blit(rendered_line, (x, current_y))
                            current_y += line_height
                            wrapped_line = word + " "
                        else:
                            wrapped_line = test_line
                    # Render any remaining text in the wrapped line
                    if wrapped_line.strip():
                        rendered_line = self.font.render(wrapped_line, True, self.color)
                        surface.blit(rendered_line, (x, current_y))
                        current_y += line_height
                    continue

            # Blit the rendered line
            surface.blit(rendered_line, (x, current_y))
            current_y += line_height
