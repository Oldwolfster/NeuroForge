import pygame
from src.NNA.Legos.Activation import get_activation_derivative_formula
from src.NNA.Legos.Optimizer import *

from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights

from src.NNA.engine.Neuron import Neuron
from src.NNA.utils.general_text import is_numeric, smart_format
from src.NNA.utils.pygame import draw_gradient_rect

from src.NeuroForge import Const
import json

from src.NeuroForge.EZPrint import EZPrint


class DisplayModel__Neuron_Base:
    """
    DisplayModel__Neuron is created by DisplayModel.
    Note: DisplayModel inherits from EzSurface, DisplayModel__Neuron does not!
    This class has the following primary purposes:
    1) Store all information related to the neuron
    2) Update that information when the sample or epoch changes.
    3) Draw the "Standard" components of the neuron.  (Body, Banner, and Banner Text)
    4) Invoke the appropriate "Visualizer" to draw the details of the Neuron
    """
    __slots__ = ("banner_text_width","on_screen", "visualizer_mode", "learning_rates","TRI","output_layer","location_right_side", "location_bottom_side","am_really_short", "model", "is_input","config", "column_widths","cached_tooltip", "text_version", "last_epoch","last_sample", "font_header", "header_text", "font_body", "max_per_weight", "max_activation",  "run_id", "screen", "db", "rs", "nid", "layer", "position", "is_output", "label", "location_left", "location_top", "location_width", "location_height", "weights", "weights_before", "neuron_inputs", "raw_sum", "activation_function", "activation_value", "blame","activation_gradient", "banner_text", "tooltip_columns", "weight_adjustments", "blame_calculations", "avg_err_sig_f6or_epoch", "loss_gradient", "ez_printer", "neuron_visualizer", "neuron_build_text" )

    input_values = []   # Class variable to store inputs #TODO Delete me
    def __repr__(self):
        """Custom representation for debugging."""
        return f"Neuron {self.label})"
    def __init__(self, model, left: int, top: int, width: int, height:int, nid: int, layer: int, position: int, output_layer: int, text_version: str,  run_id: str, screen: pygame.surface, max_activation: float, is_input = ""):
        self.run_id                 = run_id
        self.model                  = model
        self.TRI                    = self.model.TRI
        self.config                 = self.model.config
        #print (f"In DM_Neuron {self.model} ")
        self.screen                 = screen
        self.db                     = Const.dm.db
        self.rs                     = None  # Store result of querying sample/Neuron table for this sample/epoch
        self.nid                    = nid
        self.layer                  = layer
        self.position               = position
        self.output_layer           = len(self.config.architecture) - 1
        #ez_debug(out=output_layer,layer=layer)
        self.is_output              = self.output_layer == layer
        #ez_debug(self_is_out=self.is_output)
        self.max_activation         = max_activation
        self.label                  = f"{layer}-{position}"

        # Positioning
        self.location_left          = left
        self.location_top           = top
        self.location_width         = width
        self.location_height        = height
        self.location_right_side    = left + width
        self.location_bottom_side   = top + height

        # Neural properties
        self.weights                = []
        self.neuron_inputs          = []
        self.max_per_weight         = []
        self.activation_function    = self.get_activation_function()
        self.raw_sum                = 0.0
        self.activation_value       = 0.0
        self.activation_gradient    = 0.0
        self.blame                  = 0.0


        # Visualization properties
        self.am_really_short        = False
        self.banner_text            = ""
        self.tooltip_columns        = []
        self.weight_adjustments     = ""
        self.avg_err_sig_for_epoch  = 0.0
        self.loss_gradient          = 0.0
        self.neuron_build_text      = "fix me"
        self.ez_printer             = EZPrint(pygame.font.Font(None, 24), color=Const.COLOR_BLACK, max_width=200, max_height=100, sentinel_char="\n")
        self.get_max_val_per_wt()
        self.initialize_fonts()
        self.on_screen = True
        # Conditional visualizer
        self.update_neuron()        # must come before selecting visualizer
        self.neuron_visualizer      = DisplayModel__NeuronWeights(self, self.ez_printer)
        self.text_version           = text_version
        if self.is_output:
            self.banner_text = "Out"
            if self.text_version == "Verbose":
                self.banner_text = "Output Neuron"
        else:

            self.banner_text = self.label
            if self.text_version == "Verbose":
                self.banner_text = f"Hidden Neuron {self.label}"

        # Calculate and store banner text width for blame bar calculations
        self.calculate_banner_text_width()

        self.is_input              = is_input
        #ez_debug(inorout=is_input)

        self.visualizer_mode = "standard"
        self._from_base_constructor()


    def _from_base_constructor(self) -> bool:
        """Override to have code run after contstructor"""
        pass

    def get_activation_function(self) -> str:
        #print(f"actication function:  NID = {self.nid} layer = {self.layer}  Arch{len(self.TRI.config.architecture)}")
        if self.layer == -1: return "Not that kind of neuron"
        if self.layer +1 == len(self.TRI.config.architecture):  #output neuron
            return self.TRI.config.output_activation
        else:
            return self.TRI.config.hidden_activation


    def calculate_banner_text_width(self):
        """
        Calculates and stores the width of the banner text.
        Uses the same font size (30) as draw_neuron() for consistency.
        """
        font = pygame.font.Font(None, 30)
        text_surface = font.render(self.banner_text, True, Const.COLOR_FOR_NEURON_TEXT)
        self.banner_text_width = text_surface.get_width() + 15  # Add padding for safety


    def get_global_rect(self):
        return pygame.Rect(self.model.left + self.location_left,
                           self.model.top + self.location_top,
                           self.location_width,
                           self.location_height)

    def draw_neuron(self):
        """Draw the neuron visualization."""
        if not self.on_screen: return  # Don't draw if not visible
        # Font setup
        font = pygame.font.Font(None, 30) #TODO remove and use EZ_Print

        # Add 3D effect behind banner
        self.draw_top_plane()

        # Banner text
        #if self.nid == 2: print(f"self.height={self.location_height}")
        label_surface = font.render(self.banner_text, True, Const.COLOR_FOR_NEURON_TEXT)
        neuron_right_label = ""
        #        if self.location_height> 60:            neuron_right_label =  self.activation_function

        output_surface = font.render(neuron_right_label, True, Const.COLOR_FOR_NEURON_TEXT)
        label_strip_height = label_surface.get_height() + 8  # Padding

        # Draw the neuron body below the label
        self.am_really_short = self.location_height < 39# label_strip_height*2
        #ez_debug(nid=self.nid, location_height = self.location_height, label_strip_height=label_strip_height)
        if not self.am_really_short:
            body_y_start = self.location_top + label_strip_height
            body_height = self.location_height - label_strip_height
            pygame.draw.rect(self.screen,  Const.COLOR_FOR_NEURON_BODY, (self.location_left, body_y_start, self.location_width, body_height), border_radius=6, width=7)

        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        draw_gradient_rect(self.screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END, border_radius=3)
        self.screen.blit(label_surface, (self.location_left + 5, self.location_top + 5 + (label_strip_height - label_surface.get_height()) // 2))
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5
        self.screen.blit(output_surface, (right_x, self.location_top + 5 + (label_strip_height - output_surface.get_height()) // 2))

        # Render visual elements
        if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer:
            self.neuron_visualizer.render() #, self, body_y_start)

    def draw_pill(self, rect, color=None):
        """
        Draws a horizontal pill (oval the long way) into rect:
        two half-circles on the ends plus a connecting rectangle.
        """
        if color is None:
            color = Const.COLOR_BLUE_SKY

        x, y, w, h = rect
        radius = h // 2

        # center rectangle
        center_rect = pygame.Rect(x + radius, y, w - 2 * radius, h)
        pygame.draw.rect(self.screen, color, center_rect)

        # end-caps
        pygame.draw.circle(self.screen, color, (x + radius, y + radius), radius)
        pygame.draw.circle(self.screen, color, (x + w - radius, y + radius), radius)

    def draw_top_plane(self):
        """Draws the 3D looking oval behind the banner to give it depth"""
        banner_height = 40  # Should match the label_strip_height calculation
        top_plane_rect = pygame.Rect(
            self.location_left,
            self.location_top - 10,
            self.location_width,
            banner_height
        )
        self.draw_pill(top_plane_rect)

    def update_neuron(self):
        #print(f"updating neuron {self.nid}")
        if not self.update_avg_error():
            return #no record found so exit early
        if not self.on_screen: return  # Skip expensive updates if off-screen
        self.update_rs()
        self.update_weights()


    def get_max_val_per_wt(self):
        """Retrieves:The maximum absolute weight for each individual weight index across all epochs."""

        SQL_MAX_PER_WEIGHT = """                        
            SELECT MAX(ABS(value)) AS max_weight
            FROM Weight
            WHERE run_id = ? AND nid = ?
            GROUP BY weight_id
            ORDER BY weight_id ASC
        """
        max_per_weight = self.db.query_scalar_list(SQL_MAX_PER_WEIGHT, (self.run_id, self.nid))
        self.max_per_weight = max_per_weight if max_per_weight != 0 else 1

    def update_rs(self):
        # Parameterized query with placeholders
        SQL =   """
            SELECT  *
            FROM    RecordSample I
            JOIN    Neuron N
            ON      I.run_id    = N.run_id 
            AND     I.epoch     = N.epoch
            AND     I.sample_id = N.sample_id
            WHERE   N.run_id    = ? AND N.sample_id = ? AND N.epoch = ? AND nid = ?
            ORDER   BY epoch,     sample_id, N.run_id, nid 
        """
        rs = self.db.query(SQL, (self.run_id, Const.vcr.CUR_SAMPLE, self.model.display_epoch, self.nid)) # Execute query
        # ✅ Check if `rs` is empty before accessing `rs[0]`
        if not rs:
            return False  # No results found
        self.rs = rs[0]
        self.loss_gradient =  float(rs[0].get("loss_gradient", 0.0))
        self.neuron_inputs = json.loads( rs[0].get("neuron_inputs"))

        # Activation function details
        self.raw_sum                = rs[0].get('raw_sum', 0.0)
        self.activation_value       = rs[0].get('activation_value', None)        #THE OUTPUT
        self.blame                  = rs[0].get('accepted_blame', 'Unknown')  # Accepted blame
        self.activation_gradient    = rs[0].get('activation_gradient', None)  # From neuron

        lr_json = rs[0].get('learning_rates', '[]') # Extract learning rates (stored as JSON in database)
        self.learning_rates = json.loads(lr_json) if isinstance(lr_json, str) else lr_json
        #self.banner_text = f"{self.label}  Output: {smart_format( self.activation_value)}"

    def update_avg_error(self):
        SQL = """
        SELECT AVG(ABS(accepted_blame)) AS avg_accepted_blame            
        FROM Neuron
        WHERE 
        run_id   = ? and
        epoch = ? and  -- Replace with the current epoch(ChatGPT is trolling us)
        nid     = ?      
        """
        #print(f"In update_avg_error  self.model={self.model}")
        params = (self.run_id,  self.model.display_epoch, self.nid)
        rs = self.db.query(SQL, params)  # Execute query

        # ✅ Check if `rs` is empty before accessing `rs[0]`
        if not rs:
            return False  # No results found

        # ✅ Ensure `None` does not cause an error
        self.avg_err_sig_for_epoch = float(rs[0].get("avg_accepted_blame") or 0.0)
        #print("in update_avg_error returning TRUE")
        return True
    def update_weights(self):
        """Fetches weights from the Weight table instead of JSON and populates self.weights and self.weights_before."""
        SQL = """
            SELECT weight_id, value, value_before
            FROM Weight
            WHERE run_id = ? AND nid = ? AND epoch = ? AND sample = ?
            ORDER BY weight_id ASC
        """
        weights_data = self.db.query(SQL, (self.run_id, self.nid, self.model.display_epoch, Const.vcr.CUR_SAMPLE), False)
        #if self.nid == 5:
            #print(f"self.run_id={self.run_id}  self.model.display_epoch={ self.model.display_epoch}")
            #print(f"weights_data={weights_data}")

        if weights_data:
            self.weights = [column[1] for column in weights_data]  # Extract values
            self.weights_before = [column[2] for column in weights_data]  # Extract previous values
        else:
            # TODO: Handle case where no weights are found for the current epoch/sample
            self.weights = []
            self.weights_before = []

    def OLDinitialize_fonts(self):
        self.font_header            = pygame.font.Font(None, Const.TOOLTIP_FONT_TITLE)
        self.font_body              = pygame.font.Font(None, Const.TOOLTIP_FONT_BODY)
        self.header_text            = self.font_header.render("Prediction               Adjust Weights To Improve", True, Const.COLOR_BLACK)

    def initialize_fonts(self):
        self.font_header = pygame.font.Font(None, Const.TOOLTIP_FONT_TITLE)
        self.font_section = pygame.font.Font(None, Const.TOOLTIP_FONT_SUB)  # Medium for Forward/Backward
        self.font_body = pygame.font.Font(None, Const.TOOLTIP_FONT_BODY)

        # Create title based on neuron type
        if self.is_output:
            title = "Output Neuron: Prediction and Update Details"
        else:
            title = f"NEURON {self.label}: Prediction and Update Details"

        self.title_text = self.font_header.render(title, True, Const.COLOR_BLACK)
        self.header_text = self.font_section.render(" Forward Pass                  Backward Pass", True, Const.COLOR_BLACK)

############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################
############################### BELOW HERE IS POP UP WINDOW ##################################

################### All popup divider lines #############################
    def draw_all_popup_dividers(self):
        self.draw_lines_for_header(0)
        self.draw_lines_for_header(1)
        self.draw_lines_for_weighted_sum(0)
        self.draw_lines_for_weighted_sum(1)
        self.draw_lines_forward_pass_only(0)
        self.draw_lines_forward_pass_only(2)
        self.draw_popup_vertical_divider_between_forward_and_backprop()
        self.draw_highlighted_popup_cell(len(self.weights)+2, 5)
        #blame_y = 10 if self.layer == self.output_layer else 12
        #TODO Fix this self.draw_highlighted_popup_cell(len(self.weights*2)+1, blame_y)



    def draw_lines_for_header(self, extra_row : int):
        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_BLACK,
            (Const.TOOLTIP_PADDING, self.y_coord_for_row(Const.TOOLTIP_LINE_OVER_HEADER_Y + extra_row)),
            (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING, self.y_coord_for_row(Const.TOOLTIP_LINE_OVER_HEADER_Y+ extra_row)),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )

    def draw_lines_for_weighted_sum(self, extra_row : int):
        num_weights = len(self.weights)  # Includes bias and weights
        row_index = 1 + num_weights  # +1 for the header row
        y = self.y_coord_for_row(row_index + extra_row)

        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_GRAY_DARK,
            (Const.TOOLTIP_PADDING, y),
            (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING, y),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )

    def draw_lines_forward_pass_only(self, extra_row : int):
        num_weights = len(self.weights)  # Includes bias and weights
        row_index = 4 + num_weights  # +1 for the header row
        y = self.y_coord_for_row(row_index + extra_row)

        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_GRAY_DARK,
            (Const.TOOLTIP_PADDING, y),
            (self.x_coord_for_col(Const.TOOLTIP_LINE_BEFORE_BACKPROP), y),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )
    def draw_popup_vertical_divider_between_forward_and_backprop(self):
        x = self.x_coord_for_col(Const.TOOLTIP_LINE_BEFORE_BACKPROP)
        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_BLACK,
            (x, Const.TOOLTIP_HEADER_PAD),
            (x, Const.TOOLTIP_HEIGHT - Const.TOOLTIP_PADDING),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS+2
        )

    def draw_highlighted_popup_cell(self, row_index: int, col_index: int):
        x = self.x_coord_for_col(col_index)
        y = self.y_coord_for_row(row_index)
        width = self.column_widths[col_index]
        height = Const.TOOLTIP_ROW_HEIGHT

        # Draw shaded background
        pygame.draw.rect(
            self.cached_tooltip,
            Const.COLOR_HIGHLIGHT_FILL,
            pygame.Rect(x, y, width, height)
        )

        # Draw border
        pygame.draw.rect(
            self.cached_tooltip,
            Const.COLOR_HIGHLIGHT_BORDER,
            pygame.Rect(x, y, width, height),
            Const.TOOLTIP_HEADER_DIVIDER_THICKNESS
        )
    def get_title_offset(self):
        """Returns the Y offset needed to account for the title header."""
        return self.title_text.get_height() + 5

    def y_coord_for_row(self, row_index: int) -> int:
        return Const.TOOLTIP_HEADER_PAD + (row_index * Const.TOOLTIP_ROW_HEIGHT) + self.get_title_offset()

    def x_coord_for_col(self, index: int) -> int:
        return Const.TOOLTIP_PADDING + sum(self.column_widths[:index])

    def render_tooltip(self):
        """
        Render the tooltip with neuron details.
        Cache the rendered tooltip and only update if epoch or sample changes.
        """

        # ✅ Check if we need to redraw the tooltip
        if not hasattr(self, "cached_tooltip") or self.last_epoch != self.model.display_epoch or self.last_sample != Const.vcr.CUR_SAMPLE:
            self.last_epoch = self.model.display_epoch  # ✅ Update last known epoch
            self.last_sample = Const.vcr.CUR_SAMPLE  # ✅ Update last known sample

            # ✅ Tooltip dimensions
            tooltip_width = Const.TOOLTIP_WIDTH
            tooltip_height = Const.TOOLTIP_HEIGHT

            # ✅ Create a new surface for the tooltip
            self.cached_tooltip = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)

            # ✅ Fill background and draw border
            self.cached_tooltip.fill(Const.COLOR_CREAM)
            pygame.draw.rect(self.cached_tooltip, Const.COLOR_BLACK, (0, 0, tooltip_width, tooltip_height), 2)

            # ✅ Draw header
            #old self.cached_tooltip.blit(self.header_text, (Const.TOOLTIP_PADDING, Const.TOOLTIP_PADDING))
            # ✅ Draw title header
            title_y = Const.TOOLTIP_PADDING
            title_x = (Const.TOOLTIP_WIDTH - self.title_text.get_width()) // 2
            self.cached_tooltip.blit(self.title_text, (title_x, title_y+3))
            # ✅ Draw horizontal line below title
            line_y = title_y + self.title_text.get_height() + 3
            pygame.draw.line(
                self.cached_tooltip,
                Const.COLOR_BLACK,
                (Const.TOOLTIP_PADDING, line_y),
                (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING, line_y),
                4  # Line thickness
            )
            # ✅ Draw section headers (Forward Pass / Backward Pass)
            # Position it below the title with some spacing
            section_header_y = title_y + self.title_text.get_height() + 10
            self.cached_tooltip.blit(self.header_text, (Const.TOOLTIP_PADDING, section_header_y))


            # ✅ Populate content
            self.tooltip_generate_text()

            self.draw_all_popup_dividers()
            x_offset = Const.TOOLTIP_PADDING

            # ✅ Draw each column with dynamic spacing
            for col_index, (column, col_width) in enumerate(zip(self.tooltip_columns, self.column_widths)):
                for row_index, text in enumerate(column):
                    text_color = self.get_text_color(col_index, row_index, text)
                    text = self.smart_format_for_popup(text)
                    #label = self.font_body.render(str(text), True, text_color)
                    #self.cached_tooltip.blit(label, (                        x_offset,                        Const.TOOLTIP_HEADER_PAD + row_index * Const.TOOLTIP_ROW_HEIGHT + Const.TOOLTIP_PADDING                    ))
                    label = self.font_body.render(str(text), True, text_color)
                    text_rect = label.get_rect()
                    #y_pos = Const.TOOLTIP_HEADER_PAD + row_index * Const.TOOLTIP_ROW_HEIGHT + Const.TOOLTIP_PADDING
                    # Add extra offset for the title header
                    title_offset = self.title_text.get_height() + 5
                    y_pos = Const.TOOLTIP_HEADER_PAD + title_offset + row_index * Const.TOOLTIP_ROW_HEIGHT + Const.TOOLTIP_PADDING
                    x_pos = x_offset

                    if self.is_right_aligned(text, row_index):
                        text_rect.topright = (x_offset + col_width - Const.TOOLTIP_PADDING, y_pos)
                    else:
                        text_rect.topleft = (x_offset + Const.TOOLTIP_PADDING, y_pos)
                    self.cached_tooltip.blit(label, text_rect)
                x_offset += col_width  # ✅ Move X position based on column width

        # ✅ Get mouse position and adjust tooltip placement
        mouse_x, mouse_y = pygame.mouse.get_pos()
        tooltip_x = self.adjust_position(mouse_x + Const.TOOLTIP_PLACEMENT_X, Const.TOOLTIP_WIDTH, Const.SCREEN_WIDTH)
        tooltip_y = self.adjust_position(mouse_y + Const.TOOLTIP_PLACEMENT_Y, Const.TOOLTIP_HEIGHT, Const.SCREEN_HEIGHT)

        # ✅ Draw cached tooltip onto the screen
        Const.SCREEN.blit(self.cached_tooltip, (tooltip_x, tooltip_y))

    def get_text_color(self,col_index, row_index, text):
        #if col_index == 7 and row_index > 0 and text:
        if  row_index > 11111 and text:
            if is_numeric(text):
                value = float(text.replace(",", ""))
                return Const.COLOR_GREEN_FOREST if value >= 0 else Const.COLOR_CRIMSON
        return Const.COLOR_BLACK

    def adjust_position(self, position, size, screen_size):
        # If the tooltip would overflow to the right
        if position + size > screen_size:
            position = screen_size - size - Const.TOOLTIP_ADJUST_PAD

        # If the tooltip would overflow to the left
        if position < Const.TOOLTIP_ADJUST_PAD:
            position = Const.TOOLTIP_ADJUST_PAD

        return position

    def tooltip_generate_text(self):
        """Clears and regenerates tooltip text columns."""
        self.tooltip_columns.clear()
        self.tooltip_columns.extend(self.tooltip_columns_for_forward_pass())
        self.tooltip_columns.extend(self.tooltip_columns_for_backprop())  # 🔹 Uncomment when backprop data is ready
        self.set_column_widths()

    def set_column_widths(self):
        """
        Dynamically sets the column widths based on tooltip column headers.
        Assumes self.tooltip_columns is a list of vertical columns (each a list: [header, val1, val2, ...]).
        """
        col_info = 65
        col_operator = 10
        forward_cols = [45, 50, 10, col_info, 15, col_info]  # First 6 are fixed-width forward pass columns

        dynamic_widths = []
        for col in self.tooltip_columns[6:-3]:  # skip forward pass (6), and standard trailing stats (3)
            header = col[0]
            is_operator = isinstance(header, str) and len(header.strip()) <= 2 and header.strip().lower() not in {"m", "v", "t"}
            width = col_operator if is_operator else col_info
            dynamic_widths.append(width)

        last_cols = [col_info, col_info, col_info]  # Final 3 summary stats (e.g., adjustment, lr, new weight)

        self.column_widths = forward_cols + dynamic_widths + last_cols
        Const.TOOLTIP_WIDTH = sum(self.column_widths)+69

################### Gather Values for Back Pass #############################
################### Gather Values for Back Pass #############################
################### Gather Values for Back Pass #############################

    def _build_column_lists(self, headers, operators, rows):
        """
        Given:
          headers   = ['Input', 'Blame', 'Raw Adj', ...]
          operators = ['*',      '=',    '*',     ...]
          rows      = [(val1, val2, val3, ...), ...]
        returns a list-of-lists like:
          [
            ['Input',   row1[0], row2[0], …],
            ['*',       '*',     '*',     …],
            ['Blame',   row1[1], row2[1], …],
            ['=',       '=',     '=',     …],
            ['Raw Adj', row1[2], row2[2], …],
            ['*',       '*',     '*',     …],
            …
          ]
        """
        columns = []
        for h, op in zip(headers, operators):
            columns.append([h])
            columns.append([op])

        for row in rows:
            for i, val in enumerate(row):
                columns[2*i].append(val)
                # operator is constant per column
                columns[2*i+1].append(operators[i])

        return columns

    # DisplayModel__Neuron_Base.py

    def tooltip_columns_for_backprop(self):
        """Single method replaces update + finalize + standard_finale"""

        sql = """
            SELECT *
            FROM WeightAdjustments
            WHERE run_id = ?
              AND epoch = ?
              AND sample_num = ?
              AND nid = ?
            ORDER BY weight_id ASC
        """
        params = (self.run_id, self.model.display_epoch, Const.vcr.CUR_SAMPLE, self.nid)
        rows = Const.dm.db.query(sql, params, as_dict=True)

        if not rows:
            return []

        # Get column names from optimizer (or from first row)
        # Exclude the key fields - just show the interesting data
        skip_cols = {"run_id", "epoch", "sample_num", "nid", "weight_id"}
        display_cols = [k for k in rows[0].keys() if k not in skip_cols]

        # Build column lists: header + one value per weight
        result = []
        for col_name in display_cols:
            column = [col_name]  # Header
            for row in rows:
                column.append(self.smart_format_for_popup(row[col_name]))
            result.append(column)

        return result


    def tooltip_columns_for_backprop_update(self, is_batch: bool):
        # pick the right header/operator lists
        # Pick the right headers based on CURRENT display context
        if is_batch:
            headers = self.config.optimizer._backprop_popup_headers_batch
            operators = self.config.optimizer._backprop_popup_operators_batch
        else:
            headers = self.config.optimizer._backprop_popup_headers_single
            operators = self.config.optimizer._backprop_popup_operators_single

        # build the SELECT clause dynamically
        num_args   = len(headers)
        arg_fields = [f"arg_{i+1}" for i in range(num_args)]
        table      = f"WeightAdjustments_update_{self.run_id}"

        sql = f"""
            SELECT {', '.join(arg_fields)}
              FROM {table} A
             WHERE A.epoch     = ?
               AND A.sample    = ?
               AND A.nid       = ?
             ORDER BY A.weight_index ASC
        """
        params = (self.model.display_epoch,
                  Const.vcr.CUR_SAMPLE,
                  self.nid)
        rows = Const.dm.db.query(sql, params, as_dict=False)

        return self._build_column_lists(headers, operators, rows)

    def tooltip_columns_for_backprop_finalize(self, is_batch: bool):
        # pick header/operator for the JOINed finalize block
        # (you can keep this separate from standard_finale)
        # Read directly from the optimizer object
        headers = self.config.optimizer._backprop_popup_headers_finalizer
        operators = self.config.optimizer._backprop_popup_operators_finalizer

        # If optimizer has no finalizer headers, return empty
        if not headers:
            return []

        num_args   = len(headers)
        arg_fields = [f"B.arg_{i+1}" for i in range(num_args)]
        upd_table  = f"WeightAdjustments_update_{self.run_id}"
        fin_table  = f"WeightAdjustments_finalize_{self.run_id}"

        sql = f"""
            SELECT {', '.join(arg_fields)}
              FROM {upd_table} AS A
         LEFT JOIN {fin_table} AS B
                ON A.batch_id     = B.batch_id
               AND A.epoch        = B.epoch
               AND A.nid          = B.nid
               AND A.weight_index = B.weight_index
             WHERE A.epoch     = ?
               AND A.sample = ?
               AND A.nid       = ?
             ORDER BY A.weight_index ASC
        """
        params = (self.model.display_epoch,
                  Const.vcr.CUR_SAMPLE,
                  self.nid)
        rows = Const.dm.db.query(sql, params, as_dict=False)

        return self._build_column_lists(headers, operators, rows)

    def tooltip_columns_for_backprop_standard_finale(self) -> list:
        #col_lr = ["Lrn Rt"]  # NEW COLUMN
        col_delta = ["Adj"]
        col_before = ["Before"]
        col_after = ["After"]

        for i in range(len(self.weights)):
            # Get learning rate for this weight
            # weight_id 0 is bias, weights start at index 1
            weight_id = i + 1

            # Get LR from the learning_rates array
            if hasattr(self, 'learning_rates') and weight_id < len(self.learning_rates):
                lr = self.learning_rates[weight_id]
            else:
                # Fallback to config LR if learning_rates not available
                lr = self.config.learning_rate if hasattr(self.config, 'learning_rate') else 0.0

            # Calculate adjustment
            adjustment = self.weights_before[i] - self.weights[i]

            # Add values to columns
            #col_lr.append(self.smart_format_for_popup(lr))
            col_delta.append(self.smart_format_for_popup(adjustment))
            col_before.append(self.smart_format_for_popup(self.weights_before[i]))
            col_after.append(self.smart_format_for_popup(self.weights[i]))

        return [col_delta, col_before, col_after]


    def tooltip_columns_for_accepted_blame_calculation(self, all_cols):
        # Row in the box between adj and blame
        #print(f"len(all_cols)={len(all_cols)}")  #Prints blank row, empty space in each cell
        for i in range(8):  #Do entire row
            if i == 0:
                all_cols[0].append("BLAME SOURCES BELOW")
            else:
                all_cols[i].append(" ")
        #ez_debug(is_output=self.is_output)
        #ez_debug(self_layer=self.layer)
        #ez_debug(output_neuron_layer=Neuron.output_neuron.layer_id)
        if self.is_output:
        #if self.is_output: # This is an output neuron
            return self.tooltip_columns_for_error_sig_outputlayer(all_cols)
        else:
            return self.tooltip_columns_for_error_sig_hiddenlayer(all_cols)

    def tooltip_columns_for_error_sig_outputlayer(self, all_cols):
        #all_cols[0].append("Accepted Blame Calculation Below")
        all_cols[0].append("This is where blame originates and")
        all_cols[0].append("flows back through the network.")
        all_cols[0].append(" ")
        all_cols[0].append( f"Accepted Blame = Loss Gradient * Activation Gradient")
        all_cols[0].extend([f"Accepted Blame = {smart_format( self.loss_gradient)} * {smart_format(self.activation_gradient)} = {smart_format(self.loss_gradient * self.activation_gradient)}"])
        return all_cols


    def tooltip_columns_for_error_sig_hiddenlayer(self, all_cols):
        """
        Builds the bottom section showing where blame comes from (downstream neurons).

        Table structure:
        ┌──────────┬──────────┬─────────────┬───────────┐
        │ From     │ Weight   │ Their Blame │ Slice     │
        ├──────────┼──────────┼─────────────┼───────────┤
        │ 2-0      │  30.04   │   4,623     │  138,883  │
        │ 2-1      │  15.20   │   2,100     │   31,920  │
        └──────────┴──────────┴─────────────┴───────────┘
        Blame from All:                         170,803
        × Activation Gradient (fwd):              0.001
        ─────────────────────────────────────────────────
        Accepted Blame:                           170.803
        """
        col_from = 0
        col_weight = 2
        col_blame = 4
        col_slice = 8

        # Add spacing row before this section
        all_cols[col_from].append(" ")
        all_cols[col_weight].append(" ")
        all_cols[col_blame].append(" ")
        all_cols[col_slice].append(" ")
        all_cols[col_slice].append(" ")

        # Add header row
        all_cols[col_from].append("From")
        all_cols[col_weight].append(" Weight")
        all_cols[col_blame].append("Their Blame")
        all_cols[col_slice].append("     Slice")

        blame_from_all, accepted_blame = self.tooltip_columns_for_error_sig_hiddenlayer_data_rows(all_cols, col_from, col_weight, col_blame, col_slice)

        all_cols[col_from].append(" ")
        all_cols[col_weight].append(" ")
        all_cols[col_blame].append(" ")
        all_cols[col_slice].append(" ")

        # Add summary rows
        all_cols[col_from].append(" ")
        all_cols[col_weight].append(" ")
        #all_cols[col_blame].append("  Blame from All:")
        all_cols[col_blame].append("   Sum of Slices:")
        all_cols[col_slice].append(blame_from_all)

        all_cols[col_from].append("  Activation Gradient (from fwd pass):")
        all_cols[col_weight].append(" ")
        all_cols[col_blame].append(" ")
        all_cols[col_slice].append(self.activation_gradient)

        all_cols[col_from].append(" ")
        all_cols[col_weight].append(" ")
        all_cols[col_blame].append("Accepted Blame:")
        all_cols[col_slice].append(accepted_blame)

        return all_cols

    def tooltip_columns_for_error_sig_hiddenlayer_data_rows(self, all_cols, col_from, col_weight, col_blame, col_slice):
        # Get data from database
        weights, blames = self.get_elements_of_backproped_error()

        if not weights:  # No downstream connections
            return all_cols

        # Generate "From" labels for downstream neurons
        downstream_layer = self.layer + 1
        feeds_to_output = (self.layer == len(self.config.architecture) - 2)

        # Add data rows - one per downstream neuron
        for i, (weight, blame) in enumerate(zip(weights, blames)):
            # Generate label for downstream neuron
            if feeds_to_output:
                from_label = "Out" if len(weights) == 1 else f"Out-{i}"
            else:
                from_label = f"{downstream_layer}-{i}"

            # Calculate slice
            slice_value = weight * blame

            # Add row
            all_cols[col_from].append(from_label)
            all_cols[col_weight].append(weight)
            all_cols[col_blame].append(blame)
            all_cols[col_slice].append(slice_value)

        # Calculate totals
        slices = [w * b for w, b in zip(weights, blames)]
        blame_from_all = sum(slices)
        accepted_blame = blame_from_all * self.activation_gradient
        return blame_from_all, accepted_blame


    def get_elements_of_backproped_error(self):
        """Fetches elements required to calculate backpropogated error for a hidden neuron"""
        SQL = """
            SELECT arg_1, arg_2
            FROM ErrorSignalCalcs
            WHERE run_id = ? AND nid = ? AND epoch = ? AND sample = ?
            ORDER BY weight_id ASC
        """
        backprop_error_elements = self.db.query(SQL, (self.run_id, self.nid, self.model.display_epoch, Const.vcr.CUR_SAMPLE), False)

        if backprop_error_elements:             # Split the elements into two lists using the helper function
            list1, list2 = self.split_error_elements(backprop_error_elements)
            return list1, list2
        else:
            return [],[]

################### Gather Values for Forward Pass #############################
################### Gather Values for Forward Pass #############################
################### Gather Values for Forward Pass #############################
    def tooltip_columns_for_forward_pass_row_labels(self, inputs):
        labels = ["Cog", "Bias"]
        for i,inp in enumerate(inputs[:-2]):
            labels.append(f"Wt {i+1}")
        labels.append("Raw Sum")
        return labels

    def tooltip_columns_for_forward_pass(self):

        #Next we need the actual inputs.
        sample_data = Const.dm.get_sample_data(self.run_id)
        all_columns = []
        inputs          = self.tooltip_column_forward_pass_one_inputs(sample_data) #first item on the list is the literal "Input"

        row_labels = self.tooltip_columns_for_forward_pass_row_labels(inputs)
        all_columns.append(row_labels)
        all_columns.append(inputs)

        # Multiply signs
        multiply_signs = ["*", " "]   # the equals is for bias
        multiply_signs.extend(["*"] * (len(inputs)-2))
        all_columns.append(multiply_signs)
        weights=["Weight"]
        weights.extend(self.weights_before)
        #ez_debug(wt_with_lbl = weights)
        all_columns.append(weights)
        all_columns.append(["="] * (len(inputs) + 2) ) # col_op1
        #all_columns.extend([["="] * (len(inputs) + 1)," ", "=" ) # col_op1


        # weighted product
        # Slice inpts to start from the 3rd item (index 2) and wt_before to start from the 2nd item (index 1)
        inputs_sliced = inputs[2:]  # Slices from index 2 to the end
        wt_before_sliced = weights[2:]  # Slices from index 1 to the end
        products = [inp * wt for inp, wt in zip(inputs_sliced, wt_before_sliced)]
        product_col = ["Product", weights[1]]    #Label and bias
        product_col.extend(products)
        weighted_sum = sum(product_col[1:])     # Sums everything except the first element - calculate weighted sum
        product_col.append(weighted_sum)
        row_labels.append(f"{self.activation_function}({smart_format(weighted_sum)})")

        product_col.append(self.activation_value)
        all_columns.append(product_col)
        #ez_debug(wt_before = self.weights_before)
        #print(products)
        #ez_debug(all_columns_after_inputs=all_columns)

        row_labels.append("") #Blank row after output
        product_col.append("") #Blank row after output
        row_labels.append("Act Gradient")
        product_col.append(self.activation_gradient)
        row_labels.append( get_activation_derivative_formula(f"{self.activation_function}"))
        #TODO only add below if space permits
        row_labels.extend(["(How much 'Raw Sum' contri-","butes to final prediction)"])        #So, for hidden neurons, a better description might be something like:ow much the neuron's raw sum, after being transformed by its activation function, contributes to the propagation of error through the network."
        #inputs[1] = " "  #"N/A" # remove the 1 for bias
        return all_columns

    def tooltip_column_forward_pass_one_inputs(self,sample_data):
        input_col =["Input"]
        input_col.extend(self.neuron_inputs)
        return input_col

    def split_error_elements(self,elements):
        """
        Splits a list of tuples into two lists.
        Each tuple is expected to have two elements.

        Parameters:
            elements (list of tuple): List of tuples to split.

        Returns:
            tuple: A tuple containing two lists:
                   - The first list contains the first element of each tuple.
                   - The second list contains the second element of each tuple.
        """
        if elements:
            # Use zip to transpose the list of tuples and then convert each tuple to a list
            first_list, second_list = map(list, zip(*elements))
            return first_list, second_list
        return [], []

    def smart_format_for_popup(self, num):
        try:
            num = float(num)  # Ensure input is a number
        except (ValueError, TypeError):

            return str(num)  # If conversion fails, return as is

        if num == 0:
            return "0"
        #elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
        #    return f"{num:.2e}"
        elif abs(num) < 0.001:  # Use 6 decimal places for small numbers
            #formatted = f"{num:,.6f}"
            return f"{num:.1e}"
        elif abs(num) < 1:  # Use 3 decimal places for numbers less than 1
            formatted = f"{num:,.3f}"
        elif abs(num) > 1e6:  # Use 6 decimal places for small numbers
            return f"{num:.1e}"
        elif abs(num) > 1000:  # Use no decimal places for large numbers
            formatted = f"{num:,.0f}"

        else:  # Default to 2 decimal places
            formatted = f"{num:,.2f}"

        # Remove trailing zeros and trailing decimal point if necessary
        return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

    def is_right_aligned(self, text, row_index):
        if row_index == 0:
            return True
        if text == "||":
            return True
        if text == "/":
            return True
        if text == "=":
            return True
        if text == "*":
            return True
        if isinstance(text, (int, float)):
            return True
        if isinstance(text, str):
            cleaned = text.replace(",", "").strip()
            try:
                float(cleaned)
                return True
            except ValueError:
                return cleaned.upper() in ["N/A", "NONE"]
        return False

    def switch_to_compact(self):
        """Switch to compact visualizer"""
        from src.NeuroForge.DisplayModel__NeuronWeightsSmall import DisplayModel__NeuronWeightsSmall
        self.visualizer_mode = "compact"
        self.neuron_visualizer = DisplayModel__NeuronWeightsSmall(self, self.ez_printer)

    def switch_to_standard(self):
        """Switch to standard visualizer"""
        from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights
        self.visualizer_mode = "standard"
        self.neuron_visualizer = DisplayModel__NeuronWeights(self, self.ez_printer)