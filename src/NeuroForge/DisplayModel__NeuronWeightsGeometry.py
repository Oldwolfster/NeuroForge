import pygame

from src.NNA.utils.general_text import smart_format
from src.NeuroForge import Const


class DisplayModel__NeuronWeightsGeometry:
    """
    Strategy for visualizing a neuron's decision boundary.
    MVP: Only supports neurons with exactly 2 inputs.
    Draws the line w1*x1 + w2*x2 + b = 0 within the neuron box.
    """

    BANNER_HEIGHT = 29
    MARGIN = 10

    def __init__(self, neuron, ez_printer):
        self.neuron = neuron
        self.ez_printer = ez_printer
        self.my_fcking_labels = []  # Arrow target positions
        self.need_label_coord = True

    def recalculate_layout(self):
        """Called when neuron is resized."""
        self.my_fcking_labels = []
        self.need_label_coord = True

    def render(self):
        """Main render - draws decision boundary line."""
        weights = self.neuron.weights_before

        # Record arrow target positions (evenly spaced for each weight)
        if self.need_label_coord:
            self.compute_label_positions(len(weights))

        # MVP: Only handle exactly 3 weights (bias + 2 inputs)
        if len(weights) != 3:
            self.draw_fallback_text("N≠2")
            return

        bias, w1, w2 = weights[0], weights[1], weights[2]

        # Get drawable area (below banner, with margins)
        draw_left = self.neuron.location_left + self.MARGIN
        draw_top = self.neuron.location_top + self.BANNER_HEIGHT + self.MARGIN
        draw_width = self.neuron.location_width - 2 * self.MARGIN
        draw_height = self.neuron.location_height - self.BANNER_HEIGHT - 2 * self.MARGIN

        # Compute line endpoints clipped to drawable box
        endpoints = self.compute_boundary_line(w1, w2, bias,
                                               draw_left, draw_top,
                                               draw_width, draw_height)

        if endpoints:


            # Fill the two sides (behind the line)
            self._fill_decision_regions(w1, w2, bias, draw_left, draw_top, draw_width, draw_height)
            self._ensure_geometry_from_inputs()
            self._draw_points(draw_left, draw_top, draw_width, draw_height)

            p1, p2 = endpoints
            pygame.draw.line(self.neuron.screen, Const.COLOR_WHITE, p1, p2, 3)
            pygame.draw.line(self.neuron.screen, Const.COLOR_BLUE_PURE, p1, p2, 1)

        # Draw weight values as text
        self.draw_weight_labels(bias, w1, w2, draw_left, draw_top)

    def compute_label_positions(self, num_weights):
        """Compute evenly spaced positions for arrows to target."""
        start_x = self.neuron.location_left + 5
        start_y = self.neuron.location_top + self.BANNER_HEIGHT + 10
        available_height = self.neuron.location_height - self.BANNER_HEIGHT - 20

        if num_weights > 1:
            spacing = available_height / (num_weights - 1)
        else:
            spacing = 0

        for i in range(num_weights):
            y = start_y + i * spacing
            self.my_fcking_labels.append((start_x, y))

        self.need_label_coord = False

    def compute_boundary_lineOrigi(self, w1, w2, bias, box_x, box_y, box_w, box_h):
        """
        Compute where line w1*x1 + w2*x2 + b = 0 intersects the box.

        Maps box to normalized space: x1 ∈ [-1,1], x2 ∈ [-1,1]
        Returns two screen-space points, or None if line doesn't cross box.
        """
        intersections = []

        # Edge: x1 = -1  →  x2 = (w1 - b) / w2
        if abs(w2) > 1e-9:
            x2 = (w1 - bias) / w2
            if -1 <= x2 <= 1:
                intersections.append((-1, x2))

        # Edge: x1 = 1  →  x2 = (-w1 - b) / w2
        if abs(w2) > 1e-9:
            x2 = (-w1 - bias) / w2
            if -1 <= x2 <= 1:
                intersections.append((1, x2))

        # Edge: x2 = -1  →  x1 = (w2 - b) / w1
        if abs(w1) > 1e-9:
            x1 = (w2 - bias) / w1
            if -1 <= x1 <= 1:
                intersections.append((x1, -1))

        # Edge: x2 = 1  →  x1 = (-w2 - b) / w1
        if abs(w1) > 1e-9:
            x1 = (-w2 - bias) / w1
            if -1 <= x1 <= 1:
                intersections.append((x1, 1))

        # Remove duplicates (corner cases)
        unique = []
        for p in intersections:
            is_dup = any(abs(p[0] - q[0]) < 0.01 and abs(p[1] - q[1]) < 0.01 for q in unique)
            if not is_dup:
                unique.append(p)

        if len(unique) < 2:
            return None

        # Convert normalized coords to screen coords
        # x1: -1 → left edge, +1 → right edge
        # x2: -1 → bottom edge, +1 → top edge (screen Y is inverted)
        def to_screen(nx1, nx2):
            sx = box_x + (nx1 + 1) / 2 * box_w
            sy = box_y + box_h - (nx2 + 1) / 2 * box_h
            return (int(sx), int(sy))

        return (to_screen(*unique[0]), to_screen(*unique[1]))


    def compute_boundary_line(self, w1, w2, bias, box_x, box_y, box_w, box_h):
        """
        Compute where line w1*x1 + w2*x2 + b = 0 intersects the box.

        Maps box to normalized space: x1 ∈ [-1,1], x2 ∈ [-1,1]
        Returns two screen-space points, or None if line doesn't cross box.
        """
        intersections = []

        # Edge: x1 = -1  →  x2 = (w1 - b) / w2
        if abs(w2) > 1e-9:
            x2 = (w1 - bias) / w2
            if -1 <= x2 <= 1:
                intersections.append((-1, x2))

        # Edge: x1 = 1  →  x2 = (-w1 - b) / w2
        if abs(w2) > 1e-9:
            x2 = (-w1 - bias) / w2
            if -1 <= x2 <= 1:
                intersections.append((1, x2))

        # Edge: x2 = -1  →  x1 = (w2 - b) / w1
        if abs(w1) > 1e-9:
            x1 = (w2 - bias) / w1
            if -1 <= x1 <= 1:
                intersections.append((x1, -1))

        # Edge: x2 = 1  →  x1 = (-w2 - b) / w1
        if abs(w1) > 1e-9:
            x1 = (-w2 - bias) / w1
            if -1 <= x1 <= 1:
                intersections.append((x1, 1))

        # Remove duplicates (corner cases)
        unique = []
        for p in intersections:
            is_dup = any(abs(p[0] - q[0]) < 0.01 and abs(p[1] - q[1]) < 0.01 for q in unique)
            if not is_dup:
                unique.append(p)

        if len(unique) < 2:
            return None

        # NEW: choose the two intersections farthest apart (max-length segment)
        if len(unique) > 2:
            best_i, best_j = 0, 1
            best_d2 = -1.0
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    dx = unique[i][0] - unique[j][0]
                    dy = unique[i][1] - unique[j][1]
                    d2 = dx * dx + dy * dy
                    if d2 > best_d2:
                        best_d2 = d2
                        best_i, best_j = i, j
            p_a, p_b = unique[best_i], unique[best_j]
        else:
            p_a, p_b = unique[0], unique[1]

        # Convert normalized coords to screen coords
        # x1: -1 → left edge, +1 → right edge
        # x2: -1 → bottom edge, +1 → top edge (screen Y is inverted)
        def to_screen(nx1, nx2):
            sx = box_x + (nx1 + 1) / 2 * box_w
            sy = box_y + box_h - (nx2 + 1) / 2 * box_h
            return (int(sx), int(sy))

        return (to_screen(*p_a), to_screen(*p_b))

    def draw_weight_labels(self, bias, w1, w2, box_x, box_y):
        """Draw weight values as text labels."""
        font = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)

        labels = [
            (f"b={smart_format(bias)}", box_x + 5, box_y + 5),
            (f"w1={smart_format(w1)}", box_x + 5, box_y + 22),
            (f"w2={smart_format(w2)}", box_x + 5, box_y + 39),
        ]

        for text, x, y in labels:
            text_surface = font.render(text, True, Const.COLOR_WHITE)
            bg_rect = text_surface.get_rect(topleft=(x, y)).inflate(4, 2)
            pygame.draw.rect(self.neuron.screen, (0, 0, 0, 180), bg_rect)
            self.neuron.screen.blit(text_surface, (x, y))

    def draw_fallback_text(self, message):
        """Show message when boundary can't be drawn."""
        font = pygame.font.Font(None, 24)
        text = font.render(message, True, Const.COLOR_WHITE)
        cx = self.neuron.location_left + self.neuron.location_width // 2
        cy = self.neuron.location_top + self.neuron.location_height // 2
        rect = text.get_rect(center=(cx, cy))
        self.neuron.screen.blit(text, rect)

    def _clip_poly_halfplane(self, poly, w1, w2, bias, keep_positive: bool):
        """
        Sutherland–Hodgman clip of convex polygon 'poly' against the half-plane:
            w1*x + w2*y + b >= 0   (keep_positive=True)
            w1*x + w2*y + b <= 0   (keep_positive=False)
        poly points are in NORMALIZED coords (x,y in [-1,1]).
        """
        if not poly:
            return []

        def s(p):
            return w1 * p[0] + w2 * p[1] + bias

        def inside(val):
            return val >= 0 if keep_positive else val <= 0

        out = []
        prev = poly[-1]
        s_prev = s(prev)

        for curr in poly:
            s_curr = s(curr)
            prev_in = inside(s_prev)
            curr_in = inside(s_curr)

            if prev_in and curr_in:
                out.append(curr)
            elif prev_in and not curr_in:
                # leaving: add intersection
                t = s_prev / (s_prev - s_curr)
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                out.append((ix, iy))
            elif (not prev_in) and curr_in:
                # entering: add intersection + curr
                t = s_prev / (s_prev - s_curr)
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                out.append((ix, iy))
                out.append(curr)

            prev = curr
            s_prev = s_curr

        return out

    def _fill_decision_regions(self, w1, w2, bias, box_x, box_y, box_w, box_h):
        """
        Fills the two sides of the decision boundary inside the drawable box.
        Uses normalized space [-1,1] x [-1,1] to match compute_boundary_line().
        """
        # Light tints (adjust alpha to taste)
        GREEN = (0, 255, 0, 40)
        RED = (255, 0, 0, 40)

        # Local alpha surface (so per-pixel alpha works reliably)
        surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)

        # Rectangle in normalized coords (clockwise)
        rect = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

        # Clip rectangle to each half-plane
        poly_pos = self._clip_poly_halfplane(rect, w1, w2, bias, keep_positive=True)
        poly_neg = self._clip_poly_halfplane(rect, w1, w2, bias, keep_positive=False)

        def to_local(nx1, nx2):
            sx = (nx1 + 1) / 2 * box_w
            sy = box_h - (nx2 + 1) / 2 * box_h
            return (int(sx), int(sy))

        # Draw filled polys if they exist
        if len(poly_pos) >= 3:
            pygame.draw.polygon(surf, GREEN, [to_local(x, y) for x, y in poly_pos])
        if len(poly_neg) >= 3:
            pygame.draw.polygon(surf, RED, [to_local(x, y) for x, y in poly_neg])

        # Blit behind everything else
        self.neuron.screen.blit(surf, (box_x, box_y))



#########################DISPLAY POINTS IN THE NEURONS
    def _to_norm(self, v, vmin, vmax):
        """Map value in [vmin,vmax] to normalized [-1,1]."""
        if vmax == vmin:
            return 0.0
        t = (v - vmin) / (vmax - vmin)
        return t * 2.0 - 1.0

    def _to_screen_from_value(self, x, y, viewport, box_x, box_y, box_w, box_h):
        """Map (x,y) in value space to screen coords inside the box."""
        xmin, xmax, ymin, ymax = viewport
        nx = self._to_norm(x, xmin, xmax)
        ny = self._to_norm(y, ymin, ymax)

        sx = box_x + (nx + 1) / 2 * box_w
        sy = box_y + box_h - (ny + 1) / 2 * box_h
        return (int(sx), int(sy))

    def _draw_points(self, box_x, box_y, box_w, box_h):
        """
        Draw points if neuron provides:
          - neuron.geometry_points: [(x,y) or (x,y,color), ...]
          - neuron.geometry_viewport: (xmin,xmax,ymin,ymax)
        """
        if not hasattr(self.neuron, "geometry_points"):
            return
        if not hasattr(self.neuron, "geometry_viewport"):
            return

        pts = self.neuron.geometry_points
        vp  = self.neuron.geometry_viewport

        for p in pts:
            if len(p) == 2:
                x, y = p
                color = Const.COLOR_WHITE
            else:
                x, y, color = p[0], p[1], p[2]

            pos = self._to_screen_from_value(x, y, vp, box_x, box_y, box_w, box_h)
            pygame.draw.circle(self.neuron.screen, color, pos, 5)   # outer
            pygame.draw.circle(self.neuron.screen, (0,0,0), pos, 2) # inner dot for contrast


    def _ensure_geometry_from_inputs(self):
        """
        Uses self.neuron.inputs as the point(s) to plot.
        Works for hidden neurons (x1,x2) and output neuron (h1,h2) as long as
        neuron.inputs is the 2D input the neuron actually receives.
        """
        if not hasattr(self.neuron, "inputs"):
            return

        raw = self.neuron.inputs

        # Accept list/tuple like [x1,x2] or (x1,x2)
        if isinstance(raw, (list, tuple)):
            vals = list(raw)

        # Accept dict-like "json" (keeps insertion order in modern Python)
        elif isinstance(raw, dict):
            vals = list(raw.values())

        else:
            return

        # Need at least 2 numbers
        if len(vals) < 2:
            return

        x, y = vals[0], vals[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return

        # Single current-sample dot
        self.neuron.geometry_points = [(x, y, Const.COLOR_WHITE)]

        # If caller didn't set a viewport, auto-fit one around the point
        if not hasattr(self.neuron, "geometry_viewport"):
            pad = 0.25
            self.neuron.geometry_viewport = (x - pad, x + pad, y - pad, y + pad)
