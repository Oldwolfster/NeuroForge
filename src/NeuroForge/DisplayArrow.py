import math
import pygame




class DisplayArrow:
    def __init__(
        self,
        start_x_or_point,
        start_y_or_end_point,
        end_x=None,
        end_y=None,
        screen=None,
        thickness: int = 3,
        color: tuple = (0, 0, 0),
        arrow_size: int = 24,
        head_angle: float = math.pi / 9.69,
        draw_start_arrow: bool = False,
        thickness_offsets: tuple = (16, 8, 0),
        layer_colors: tuple = None,
        label_text=None,
        label_font=None,
        label_font_size=14,
        label_text_color=(0, 0, 0),
        label_bg_color=(255, 255, 255),
        label_padding=1,
        label_position=0.3,
        shaft_extension: int = 4,
    ):
        # Auto-detect format: points vs individual coords
        if isinstance(start_x_or_point, tuple) and isinstance(start_y_or_end_point, tuple):
            # Point format: DisplayArrow((x1,y1), (x2,y2), screen=...)
            start_x, start_y = start_x_or_point
            end_x, end_y = start_y_or_end_point
            if screen is None:
                screen = end_x  # screen is 3rd positional arg in point format
        else:
            # Individual coords format: DisplayArrow(x1, y1, x2, y2, screen=...)
            start_x = start_x_or_point
            start_y = start_y_or_end_point
            # end_x, end_y, screen already set from parameters

        # Validate all coordinates are numbers
        if not all(isinstance(v, (int, float)) for v in (start_x, start_y, end_x, end_y)):
            raise TypeError(
                f"Expected int or float, but got: "
                f"start_x={start_x}, start_y={start_y}, "
                f"end_x={end_x}, end_y={end_y}"
            )

        self.screen = screen
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

        self.screen = screen
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

        # ← existing defaults
        self.thickness = thickness
        self.color = color
        self.arrow_size = arrow_size
        self.head_angle = head_angle
        self.draw_start_arrow = draw_start_arrow

        # Label parameters
        self.label_text = label_text
        self.label_font = label_font
        self.label_font_size = label_font_size
        self.label_text_color = label_text_color
        self.label_bg_color = label_bg_color
        self.label_padding = label_padding
        self.label_position = label_position

        # ── new── how many pixels closer the shaft reaches
        self.shaft_extension = shaft_extension

        # ← layering params (optional)
        self.thickness_offsets = thickness_offsets
        self.layer_colors = tuple(layer_colors) if layer_colors is not None else None

    def draw(self):
        """ Draws either a single-color arrow (default) or a multi-layer arrow,
            with the shaft ending at the base of the arrowhead.
        """
        if self.layer_colors and len(self.layer_colors) == len(self.thickness_offsets):
            # layered mode
            for offset, col in zip(self.thickness_offsets, self.layer_colors):
                w = self.thickness + offset
                size = self.arrow_size + offset

                # ── use the extended‐shaft helper
                bx, by = self._calculate_extended_arrow_base(
                    self.end_x, self.end_y, self.start_x, self.start_y, size
                )
                pygame.draw.line(
                    self.screen, col,
                    (self.start_x, self.start_y),
                    (bx, by),
                    w,
                )

                # head at end
                pts_end = self._calculate_arrowhead(
                    self.end_x, self.end_y, self.start_x, self.start_y,
                    arrow_size=size
                )
                pygame.draw.polygon(self.screen, col, [(self.end_x, self.end_y)] + pts_end)

                # optional head at start
                if self.draw_start_arrow:
                    sx, sy = self._calculate_extended_arrow_base(
                        self.start_x, self.start_y, self.end_x, self.end_y, size
                    )
                    pygame.draw.line(
                        self.screen, col,
                        (self.end_x, self.end_y),
                        (sx, sy),
                        w,
                    )
                    pts_start = self._calculate_arrowhead(
                        self.start_x, self.start_y, self.end_x, self.end_y,
                        arrow_size=size
                    )
                    pygame.draw.polygon(self.screen, col, [(self.start_x, self.start_y)] + pts_start)

        else:
            # single-color mode
            # ── use the extended‐shaft helper
            bx, by = self._calculate_extended_arrow_base(
                self.end_x, self.end_y, self.start_x, self.start_y, self.arrow_size
            )
            pygame.draw.line(
                self.screen,
                self.color,
                (self.start_x, self.start_y),
                (bx, by),
                self.thickness,
            )

            # head at end
            pts = self._calculate_arrowhead(
                self.end_x, self.end_y, self.start_x, self.start_y
            )
            pygame.draw.polygon(self.screen, self.color, [(self.end_x, self.end_y)] + pts)

            # optional head at start
            if self.draw_start_arrow:
                sx, sy = self._calculate_extended_arrow_base(
                    self.start_x, self.start_y, self.end_x, self.end_y, self.arrow_size
                )
                pygame.draw.line(
                    self.screen,
                    self.color,
                    (self.end_x, self.end_y),
                    (sx, sy),
                    self.thickness,
                )
                pts_start = self._calculate_arrowhead(
                    self.start_x, self.start_y, self.end_x, self.end_y
                )
                pygame.d6raw.polygon(self.screen, self.color, [(self.start_x, self.start_y)] + pts_start)
        self.draw_label()

    def draw_label(self):
        """Draws text label at the midpoint of the arrow with auto-sized background."""
        if self.label_text is None:
            return

        # Calculate midpoint
        label_x = self.start_x + (self.end_x - self.start_x) * self.label_position
        label_y = self.start_y + (self.end_y - self.start_y) * self.label_position

        # Get or create font
        if self.label_font is not None:
            font = self.label_font
        else:
            font = pygame.font.Font(None, self.label_font_size)

        # Render text to get dimensions
        text_surface = font.render(self.label_text, True, self.label_text_color)
        text_rect = text_surface.get_rect()

        # Create background rect with padding
        bg_rect = pygame.Rect(
            0, 0,
            text_rect.width + (self.label_padding * 2),
            text_rect.height + (self.label_padding * 2)
        )
        bg_rect.center = (label_x, label_y)

        # Draw background
        pygame.draw.rect(self.screen, self.label_bg_color, bg_rect)

        # Draw text centered on background
        text_rect.center = (label_x, label_y)
        self.screen.blit(text_surface, text_rect)

    def _calculate_arrowhead(
        self,
        tip_x,
        tip_y,
        base_x,
        base_y,
        arrow_size: float = None,
        head_angle: float = None,
    ):
        """ Calculates the two base points of the triangular arrowhead. """
        size = arrow_size if arrow_size is not None else self.arrow_size
        angle_offset = head_angle if head_angle is not None else self.head_angle
        angle = math.atan2(tip_y - base_y, tip_x - base_x)

        p1 = (
            tip_x - size * math.cos(angle - angle_offset),
            tip_y - size * math.sin(angle - angle_offset),
        )
        p2 = (
            tip_x - size * math.cos(angle + angle_offset),
            tip_y - size * math.sin(angle + angle_offset),
        )
        return [p1, p2]

    def _calculate_arrow_base(self, tip_x, tip_y, base_x, base_y, arrow_size):
        """ Returns the point on the shaft where the arrowhead begins. """
        angle = math.atan2(tip_y - base_y, tip_x - base_x)
        bx = tip_x - arrow_size * math.cos(angle)
        by = tip_y - arrow_size * math.sin(angle)
        return bx, by

    def _calculate_extended_arrow_base(self, tip_x, tip_y, base_x, base_y, arrow_size):
        """ Like _calculate_arrow_base but moves the base 'shaft_extension' pixels closer to the tip. """
        effective_size = max(arrow_size - self.shaft_extension, 0)
        return self._calculate_arrow_base(tip_x, tip_y, base_x, base_y, effective_size)
