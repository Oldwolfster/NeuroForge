from typing import Tuple, List
import pygame
from src.NNA.engine.Config import Config
from src.NeuroForge import Const


class Popup_Base:
    def __init__(self,  text_color = Const.COLOR_BLACK,  highlight_differences: bool = True, column_width_overrides: dict[int, int] = None ):
        self.text_color             = text_color
        self.cached_surf            = None
        self.last_state             = None
        self.highlight_differences  = highlight_differences

        self.column_width_overrides = column_width_overrides or {} # maps col_index → forced width in px
        self.font_header            = pygame.font.Font(None, Const.TOOLTIP_FONT_HEADER)
        self.font_body              = pygame.font.Font(None, Const.TOOLTIP_FONT_BODY)
        self.font_title             = pygame.font.Font(None, Const.TOOLTIP_FONT_TITLE)

    def show_me(self):       #Actually draws it, MUST be last so need to  store delegate and run after all other drawing
        state = (Const.vcr.CUR_EPOCH_MASTER, Const.vcr.CUR_ITERATION)
        if state != self.last_state:
            self.last_state = state
            self.cached_surf = self._draw()

        # ✅ Get mouse position and adjust tooltip placement
        mouse_x, mouse_y = pygame.mouse.get_pos()
        tooltip_x = self.adjust_position(mouse_x + Const.TOOLTIP_PLACEMENT_X, Const.TOOLTIP_WIDTH, Const.SCREEN_WIDTH)
        tooltip_y = self.adjust_position(mouse_y + Const.TOOLTIP_PLACEMENT_Y, Const.TOOLTIP_HEIGHT, Const.SCREEN_HEIGHT)
        #print(f"Blitting popup at {tooltip_x},{tooltip_y} size={self.cached_surf.get_size()}")

        # ✅ Draw cached tooltip onto the screen
        Const.SCREEN.blit(self.cached_surf, (tooltip_x, tooltip_y))


    # ——— hooks for subclasses to override ———

    def is_header_cell(self, col_index, row_index) -> bool:
        """Override to change which cells are displayed as headers"""
        return col_index == 0 or row_index == 0

    def popup_max_size(self) -> Tuple[int,int]:
        """
        The absolute maximum tooltip size (e.g. to avoid overflowing the screen).
        """
        return Const.TOOLTIP_WIDTH_MAX, Const.TOOLTIP_HEIGHT_MAX

    def header_text(self) -> str:
        raise NotImplementedError

    def content_to_display(self) -> List[List[str]]:
        #def content_to_display(self) -> List[List[str]]:
        """Content for the popup as a list of columns"""
        raise NotImplementedError

    def draw_dividers(self, surf: pygame.Surface, col_widths: List[int]):
        pass

    def draw_highlights(self, surf: pygame.Surface, col_widths: List[int]):
        pass

    def right_align(self,column_index, row_index, txt): #overwrite in child class
        return False

    # ——— shared implementations ———
    def calc_widths(self, columns: List[List[str]]) -> List[int]:
        """
        1) Measure every column’s natural width (text + padding)
        2) Apply any overrides from self.column_width_overrides
        """
        pad = Const.TOOLTIP_PADDING
        measured = []
        for col in columns:
            max_w = 0
            for cell in col:
                w = self.font_body.render(str(cell), True, Const.COLOR_BLACK).get_width()
                if w > max_w:
                    max_w = w
            measured.append(max_w + pad * 2)

        # debug: see how many columns and what their natural widths are
        #ez_debug(measured_cols= measured)

        # now clamp any you asked for
        final = []
        for idx, w in enumerate(measured):
            if idx in self.column_width_overrides:
                forced = self.column_width_overrides[idx]
                #ez_debug(override_col=idx ,forced=forced)
                final.append(forced)
            else:
                final.append(w)

        #ez_debug(calc_widths_final=final)
        return final

    def _draw_cells(self, surf: pygame.Surface, columns: List[List[str]], col_widths: List[int]):
        x = Const.TOOLTIP_PADDING
        self.column_widths = col_widths  # Needed for x_coord_for_col

        for ci, col in enumerate(columns):
            for ri, txt in enumerate(col):
                is_header = self.is_header_cell(ci, ri)

                font = self.font_header if is_header else self.font_body
                label = font.render(str(txt), True, self.text_color)
                rect = label.get_rect()
                y = Const.TOOLTIP_HEADER_PAD + ri * Const.TOOLTIP_ROW_HEIGHT + Const.TOOLTIP_PADDING

                if self.right_align(ci, ri, txt):
                    rect.topright = (x + col_widths[ci] - Const.TOOLTIP_PADDING, y)
                else:
                    rect.topleft = (x + Const.TOOLTIP_PADDING, y)

                surf.blit(label, rect)

            x += col_widths[ci]

        #TODO Not seeing affect of below
        if is_header and ci == 0 and ri != 0:
            bg_rect = pygame.Rect(
                x,
                Const.TOOLTIP_HEADER_PAD + ri * Const.TOOLTIP_ROW_HEIGHT,
                col_widths[ci],
                Const.TOOLTIP_ROW_HEIGHT
            )
            pygame.draw.rect(surf, Const.COLOR_YELLOW_BRIGHT, bg_rect)

    def draw_highlights(self, surf: pygame.Surface, col_widths: List[int]):
        pass
        # highlight any row where the model-columns (cols 2+) aren’t all the same
        if not self.highlight_differences:
            return

        columns = getattr(self, "_columns", None)
        if not columns:
            return

        num_cols = len(columns)
        num_rows = len(columns[0])
        # skip header row (0) and the first two label-columns (0 & 1)
        for ri in range(1, num_rows):
            vals = [columns[c][ri] for c in range(2, num_cols)]
            # if they’re not all identical, highlight each cell
            if not all(v == vals[0] for v in vals):
                for c in range(2, num_cols):
                    x = self.x_coord_for_col(c)
                    y = self.y_coord_for_row(ri)
                    w = col_widths[c]
                    h = Const.TOOLTIP_ROW_HEIGHT
                    pygame.draw.rect(surf, Const.COLOR_YELLOW_BRIGHT, (x, y, w, h))

    def y_coord_for_row(self, row_index: int) -> int:
        return Const.TOOLTIP_HEADER_PAD + (row_index * Const.TOOLTIP_ROW_HEIGHT)

    def x_coord_for_col(self, index: int) -> int:
        return Const.TOOLTIP_PADDING + sum(self.column_widths[:index])

    def adjust_position(self, position, size, screen_size):
        # If the tooltip would overflow to the right

        if position + size > screen_size:
            position = screen_size - size - Const.TOOLTIP_ADJUST_PAD

        # If the tooltip would overflow to the left
        if position < Const.TOOLTIP_ADJUST_PAD:
            position = Const.TOOLTIP_ADJUST_PAD

        return position

    def _draw(self) -> pygame.Surface:
        w, h = self.popup_size()
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(Const.COLOR_CREAM)
        pygame.draw.rect(surf, Const.COLOR_BLACK, (0, 0, w, h), 2)

        # header
        header_text = self.header_text()
        header_surf = self.font_header.render(header_text, True, Const.COLOR_BLACK)
        surf.blit(header_surf, (Const.TOOLTIP_PADDING, Const.TOOLTIP_PADDING))

        # build table data
        columns = self.content_to_display()
        col_widths = self.calc_widths(columns) # stash for draw_highlights
        self._columns = columns

        # draw text cells
        self._draw_cells(surf, columns, col_widths)

        # draw subclass-specific lines/highlights
        self.draw_dividers(surf, col_widths)
        self.draw_highlights(surf, col_widths)
        self._draw_cells(surf, columns, col_widths) #Repeat to show over the highlights

        return surf

    def popup_size(self) -> Tuple[int,int]:
        """
        Dynamically compute tooltip size from content:
        - width = sum of all column widths
        - height = header area + row_count * row_height + bottom padding
        Then clamp to popup_max_size.
        """
        # 1) figure out what we’re actually showing
        columns = self.content_to_display()
        col_widths = self.calc_widths(columns)

        # 2) total width is just all columns side by side
        width = sum(col_widths) + Const.TOOLTIP_PADDING

        # 3) how many rows? (assume every column has same length)
        rows = len(columns[0]) if columns else 0

        # 4) height = header pad + rows*row_height + bottom padding
        height = Const.TOOLTIP_HEADER_PAD \
               + rows * Const.TOOLTIP_ROW_HEIGHT \
               + Const.TOOLTIP_PADDING

        # 5) clamp to max
        max_w, max_h = self.popup_max_size()
        return min(width,  max_w), min(height, max_h)
