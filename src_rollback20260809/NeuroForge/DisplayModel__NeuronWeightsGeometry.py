# ═══════════════════════════════════════════════════════════════════════════════
# DisplayModel__NeuronWeightsGeometry.py    COMPLETE FILE — replaces all 395 lines
#
# Contract preserved: render(), recalculate_layout(), my_fcking_labels
# Killed: compute_boundary_lineOrigi (dead), compute_boundary_line (derived from
#         the clip instead), triplicate coordinate maps, 4-arg box passing,
#         the N≠2 wall, all underscore prefixes.
# Deferred: point plotting — viewport semantics unresolved, own anchor.
# ═══════════════════════════════════════════════════════════════════════════════

import pygame

from src.NNA.utils.general_text import smart_format
from src.NeuroForge import Const


class DisplayModel__NeuronWeightsGeometry:
    """
    Renders a neuron's decision boundary as a line carving a 2D plane.

    Any neuron with 2+ weights works. Two weights are chosen as the axes
    (ranked by movement or by blame across the run, cached); every remaining
    weight folds into the constant term for the current sample, so the line
    is a true slice through the neuron's actual hyperplane.
    """

    BANNER_HEIGHT   = 29
    MARGIN          = 10
    LABEL_PITCH     = 17
    ON_LINE_EPS     = 1e-6
    TINT_POSITIVE   = (0, 255, 0, 40)
    TINT_NEGATIVE   = (255, 0, 0, 40)
    LABEL_BACKDROP  = (0, 0, 0)
    UNIT_BOX        = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    AXIS_CACHE      = {}                    # (run_id, nid, mode) -> (ia, ib) | None

    def __init__(self, neuron, ez_printer):
        self.neuron             = neuron
        self.ez_printer         = ez_printer
        self.my_fcking_labels   = []        # Arrow target positions - PUBLIC, read by DisplayModel
        self.need_label_coord   = True
        self.mode               = "movement"
        self.geo_wt_a           = None
        self.geo_wt_b           = None
        self.box                = (0, 0, 0, 0)

    def recalculate_layout(self):
        """Called when neuron is resized."""
        self.my_fcking_labels   = []
        self.need_label_coord   = True

# ══════════════════════════════════ RENDER ═══════════════════════════════════

    def render(self):
        weights = self.neuron.weights_before

        if self.need_label_coord and weights:
            self.compute_label_positions(len(weights))

        if not self.ready_to_draw(weights):
            return

        self.measure_box()
        plane = self.build_plane(weights)
        self.draw_plane(plane)
        self.draw_weight_labels(plane)

    def ready_to_draw(self, weights):
        """Axes must be pickable and present in this sample's weight list."""
        if not self.pick_geometry_weights():
            self.draw_fallback_text("N<2")
            return False
        return bool(weights) and self.geo_wt_b < len(weights)

    def measure_box(self):
        """Drawable area: below the banner, inside the margins."""
        n        = self.neuron
        self.box = (n.location_left   + self.MARGIN,
                    n.location_top    + self.BANNER_HEIGHT + self.MARGIN,
                    n.location_width  - 2 * self.MARGIN,
                    n.location_height - self.BANNER_HEIGHT - 2 * self.MARGIN)

    def draw_plane(self, plane):
        """Fill both sides, then lay the boundary over the seam."""
        keep_pos = self.clip_unit_box(plane, True)
        keep_neg = self.clip_unit_box(plane, False)
        self.fill_regions(keep_pos, keep_neg)
        self.draw_boundary(plane, keep_pos)

# ═══════════════════════════════ PLANE MATH ══════════════════════════════════

    def build_plane(self, weights):
        """(w_a, w_b, constant) - the line is w_a*x + w_b*y + constant = 0."""
        return (weights[self.geo_wt_a],
                weights[self.geo_wt_b],
                self.effective_bias(weights))

    def effective_bias(self, weights):
        """Every off-axis weight times its input. Bias included - its input is 1."""
        inputs = self.neuron.neuron_inputs
        if len(inputs) < len(weights):
            return weights[0]
        return sum(weights[i] * inputs[i]
                   for i in range(len(weights))
                   if i not in (self.geo_wt_a, self.geo_wt_b))

    def plane_value(self, plane, point):
        """Signed distance-ish. Zero means the point sits on the boundary."""
        w_a, w_b, constant = plane
        return w_a * point[0] + w_b * point[1] + constant

    def is_inside(self, value, keep_positive):
        return value >= 0 if keep_positive else value <= 0

    def crossing(self, prev, curr, s_prev, s_curr):
        """Where segment prev->curr punches through the boundary."""
        t = s_prev / (s_prev - s_curr)
        return (prev[0] + t * (curr[0] - prev[0]),
                prev[1] + t * (curr[1] - prev[1]))

# ═══════════════════════════════ CLIPPING ════════════════════════════════════

    def clip_unit_box(self, plane, keep_positive):
        """Sutherland-Hodgman: the unit box trimmed to one side of the boundary."""
        out  = []
        prev = self.UNIT_BOX[-1]
        for curr in self.UNIT_BOX:
            self.clip_edge(plane, keep_positive, prev, curr, out)
            prev = curr
        return out

    def clip_edge(self, plane, keep_positive, prev, curr, out):
        """If the edge crosses, keep the crossing. If it ends inside, keep the end."""
        s_prev  = self.plane_value(plane, prev)
        s_curr  = self.plane_value(plane, curr)
        in_prev = self.is_inside(s_prev, keep_positive)
        in_curr = self.is_inside(s_curr, keep_positive)

        if in_prev != in_curr:
            out.append(self.crossing(prev, curr, s_prev, s_curr))
        if in_curr:
            out.append(curr)

# ═══════════════════════════════ AXIS PICKING ════════════════════════════════

    def pick_geometry_weights(self):
        """Two weight ids become the axes. Cached so skin swaps never re-query."""
        if self.geo_wt_a is not None:
            return True

        key = (self.neuron.run_id, self.neuron.nid, self.mode)
        if key not in self.AXIS_CACHE:
            self.AXIS_CACHE[key] = self.rank_top_two()

        pair = self.AXIS_CACHE[key]
        if pair:
            self.geo_wt_a, self.geo_wt_b = pair
        return bool(pair)

    def rank_top_two(self):
        """Winners sorted ascending so the horizontal axis is always the lower id."""
        ranked = self.rank_by_blame() if self.mode == "blame" else self.rank_by_movement()
        if len(ranked) < 2:
            return None
        return tuple(sorted(ranked[:2]))

    def rank_by_movement(self):
        """Weight ids by distance travelled across the run. Bias excluded."""
        SQL = """
            SELECT   weight_id
            FROM     Weight
            WHERE    run_id = ? AND nid = ? AND weight_id > 0
            GROUP BY weight_id
            ORDER BY MAX(value) - MIN(value) DESC
        """
        return self.run_ranking(SQL)

    def rank_by_blame(self):
        """Weight ids by average blame magnitude across the run. Bias excluded."""
        SQL = """
            SELECT   weight_id
            FROM     ErrorSignalCalcs
            WHERE    run_id = ? AND nid = ? AND weight_id > 0
            GROUP BY weight_id
            ORDER BY AVG(ABS(result)) DESC
        """
        return self.run_ranking(SQL)

    def run_ranking(self, SQL):
        return self.neuron.db.query_column(SQL, (self.neuron.run_id, self.neuron.nid))

# ═══════════════════════════════ COORDINATES ═════════════════════════════════

    def to_local(self, point):
        """Normalized [-1,1] to pixels relative to the box. Screen Y is inverted."""
        box_x, box_y, box_w, box_h = self.box
        return (int((point[0] + 1) / 2 * box_w),
                int(box_h - (point[1] + 1) / 2 * box_h))

    def to_screen(self, point):
        """Normalized [-1,1] to absolute screen pixels."""
        box_x, box_y, box_w, box_h = self.box
        local_x, local_y           = self.to_local(point)
        return (box_x + local_x, box_y + local_y)

# ═══════════════════════════════ DRAWING ═════════════════════════════════════

    def draw_boundary(self, plane, poly):
        """The boundary IS the clipped vertices that landed on the plane."""
        on_line = [p for p in poly if abs(self.plane_value(plane, p)) < self.ON_LINE_EPS]
        if len(on_line) < 2:
            return

        point_1 = self.to_screen(on_line[0])
        point_2 = self.to_screen(on_line[-1])
        pygame.draw.line(self.neuron.screen, Const.COLOR_WHITE,     point_1, point_2, 3)
        pygame.draw.line(self.neuron.screen, Const.COLOR_BLUE_PURE, point_1, point_2, 1)

    def fill_regions(self, poly_pos, poly_neg):
        """Tint both sides on a scratch surface so per-pixel alpha behaves."""
        box_x, box_y, box_w, box_h = self.box
        surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        self.fill_one(surf, poly_pos, self.TINT_POSITIVE)
        self.fill_one(surf, poly_neg, self.TINT_NEGATIVE)
        self.neuron.screen.blit(surf, (box_x, box_y))

    def fill_one(self, surf, poly, tint):
        if len(poly) < 3:
            return
        pygame.draw.polygon(surf, tint, [self.to_local(p) for p in poly])

# ═══════════════════════════════ TEXT ════════════════════════════════════════

    def draw_weight_labels(self, plane):
        """Names which two weights are on the axes - the slice is not the whole story."""
        w_a, w_b, constant = plane
        self.draw_text_stack([f"c  ={smart_format(constant)}",
                              f"w{self.geo_wt_a} ={smart_format(w_a)}",
                              f"w{self.geo_wt_b} ={smart_format(w_b)}"])

    def draw_text_stack(self, lines):
        box_x, box_y, box_w, box_h = self.box
        font = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)
        for i, text in enumerate(lines):
            self.draw_label(font, text, box_x + 5, box_y + 5 + i * self.LABEL_PITCH)

    def draw_label(self, font, text, x, y):
        surface = font.render(text, True, Const.COLOR_WHITE)
        backing = surface.get_rect(topleft=(x, y)).inflate(4, 2)
        pygame.draw.rect(self.neuron.screen, self.LABEL_BACKDROP, backing)
        self.neuron.screen.blit(surface, (x, y))

    def draw_fallback_text(self, message):
        """Shown when the neuron has under 2 weights to plot."""
        font   = pygame.font.Font(None, 24)
        text   = font.render(message, True, Const.COLOR_WHITE)
        center = (self.neuron.location_left + self.neuron.location_width  // 2,
                  self.neuron.location_top  + self.neuron.location_height // 2)
        self.neuron.screen.blit(text, text.get_rect(center=center))

# ═══════════════════════════════ ARROW TARGETS ═══════════════════════════════

    def compute_label_positions(self, num_weights):
        """Evenly spaced arrow targets, one per weight. PUBLIC contract."""
        start_x          = self.neuron.location_left + 5
        start_y          = self.neuron.location_top  + self.BANNER_HEIGHT + 10
        available_height = self.neuron.location_height - self.BANNER_HEIGHT - 20
        spacing          = available_height / (num_weights - 1) if num_weights > 1 else 0

        for i in range(num_weights):
            self.my_fcking_labels.append((start_x, start_y + i * spacing))

        self.need_label_coord = False