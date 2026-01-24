import pygame
from src.NNA.legos.Activation import get_activation_derivative_formula
from src.NNA.legos.Optimizer import *

from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights

from src.NNA.engine.Neuron import Neuron
from src.NNA.utils.general_text import is_numeric, smart_format
from src.NNA.utils.pygame import draw_gradient_rect

from src.NeuroForge import Const
import json

from src.NeuroForge.DisplayModel__NeuronWeightsGeometry import DisplayModel__NeuronWeightsGeometry
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
        self.neuron_visualizer      = DisplayModel__NeuronWeights           (self, self.ez_printer)
        #self.neuron_visualizer      = DisplayModel__NeuronWeightsGeometry   (self, self.ez_printer)
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
            ON      I.run_id  = N.run_id 
            AND     I.epoch     = N.epoch
            AND     I.sample_id = N.sample_id
            WHERE   N.run_id = ? AND N.sample_id = ? AND N.epoch = ? AND nid = ?
            ORDER BY epoch, sample_id, N.run_id, nid 
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
        SELECT AVG(ABS(accepted_blame)) AS average_accepted_blame            
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
        self.avg_err_sig_for_epoch = float(rs[0].get("average_accepted_blame") or 0.0)
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




    def initialize_fonts(self):
        self.font_header = pygame.font.Font(None, Const.TOOLTIP_FONT_TITLE)
        self.font_section = pygame.font.Font(None, Const.TOOLTIP_FONT_SUB)
        self.font_body = pygame.font.Font(None, Const.TOOLTIP_FONT_BODY)



        if self.is_output:
            eatme = "Output Neuron"
        else:
            eatme = f"Neuron {self.label}"

        # Create centered title
        self.title_center =  f"{eatme}  Prediction and Update Details"

        # Left side text: "Output" for output neurons, "NEURON X-Y" for hidden
        if self.is_output:
            self.title_right = ""
        else:
            self.title_right = ""

        # Render parts
        self.title_center_text = self.font_header.render(self.title_center, True, Const.COLOR_BLACK)
        self.title_right_text = self.font_header.render(self.title_right, True, Const.COLOR_BLACK)

        # Build backward pass text with formula
        backward_text =  "Backward Pass1"
        if hasattr(self.TRI.config, 'optimizer') and hasattr(self.TRI.config.optimizer, 'formula') and self.TRI.config.optimizer.formula:
            backward_text += f": {self.TRI.config.optimizer.formula}"

        self.header_text = self.font_section.render(f" Forward Pass                           {backward_text}", True,
                                                    Const.COLOR_BLACK)



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


    def draw_lines_for_header(self, extra_row: int):
        pygame.draw.line(
            self.cached_tooltip,
            Const.COLOR_BLACK,
            (Const.TOOLTIP_PADDING, self.y_coord_for_row(Const.TOOLTIP_LINE_OVER_HEADER_Y + extra_row)),
            (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING +52,
             self.y_coord_for_row(Const.TOOLTIP_LINE_OVER_HEADER_Y + extra_row)),
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
            (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING+50, y),
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
        return self.title_right_text.get_height() + 5

    def y_coord_for_row(self, row_index: int) -> int:
        return Const.TOOLTIP_HEADER_PAD + (row_index * Const.TOOLTIP_ROW_HEIGHT) + self.get_title_offset()

    def x_coord_for_col(self, index: int) -> int:
        return Const.TOOLTIP_PADDING + sum(self.column_widths[:index])

    # DisplayModel__Neuron_Base.py
    def render_tooltip(self):
        """Render the tooltip with neuron details. Cache and only update if epoch or sample changes."""
        if self.should_redraw_tooltip():
            self.last_epoch = self.model.display_epoch
            self.last_sample = Const.vcr.CUR_SAMPLE
            self.rebuild_tooltip_surface()

        self.position_and_blit_tooltip()

    def should_redraw_tooltip(self):
        """Check if tooltip needs to be redrawn."""
        return (not hasattr(self, "cached_tooltip") or
                self.last_epoch != self.model.display_epoch or
                self.last_sample != Const.vcr.CUR_SAMPLE)

    def rebuild_tooltip_surface(self):
        """Rebuild the entire tooltip surface."""
        self.cached_tooltip = pygame.Surface((Const.TOOLTIP_WIDTH, Const.TOOLTIP_HEIGHT), pygame.SRCALPHA)
        self.cached_tooltip.fill(Const.COLOR_CREAM)
        pygame.draw.rect(self.cached_tooltip, Const.COLOR_BLACK, (0, 0, Const.TOOLTIP_WIDTH, Const.TOOLTIP_HEIGHT), 2)

        self.draw_tooltip_title()
        self.draw_tooltip_section_headers()
        self.tooltip_generate_text()
        self.draw_all_popup_dividers()
        self.draw_tooltip_columns()
        print("draw blame soon")
        self.draw_blame_sources_section()


    def rebuild_tooltip_surface(self):
        """Rebuild the entire tooltip surface."""
        self.cached_tooltip = pygame.Surface((Const.TOOLTIP_WIDTH, Const.TOOLTIP_HEIGHT), pygame.SRCALPHA)
        self.cached_tooltip.fill(Const.COLOR_CREAM)
        pygame.draw.rect(self.cached_tooltip, Const.COLOR_BLACK, (0, 0, Const.TOOLTIP_WIDTH, Const.TOOLTIP_HEIGHT), 2)

        self.draw_tooltip_title()
        self.tooltip_generate_text()  # <- MOVE THIS UP (creates column_widths)
        self.draw_tooltip_section_headers()  # <- NOW this can use column_widths
        self.draw_all_popup_dividers()
        self.draw_tooltip_columns()
        self.draw_blame_sources_section()

    # DisplayModel__Neuron_Base.py
    def draw_tooltip_title(self):
        """Draw the three-part title header."""
        title_y = Const.TOOLTIP_PADDING

        # Left: Neuron ID
        self.cached_tooltip.blit(self.title_right_text, (Const.TOOLTIP_PADDING + 50, title_y + 3))

        # Center: Main title
        title_x_center = (Const.TOOLTIP_WIDTH - self.title_center_text.get_width()) // 2
        self.cached_tooltip.blit(self.title_center_text, (title_x_center, title_y + 3))

        # Right: Epoch/Sample
        #title_right = self.font_header.render(f"Epoch:{self.model.display_epoch} Sample:{Const.vcr.CUR_SAMPLE}", True,
        #                                      Const.COLOR_BLACK)
        #title_x_right = Const.TOOLTIP_WIDTH - title_right.get_width() - Const.TOOLTIP_PADDING - 50
        #self.cached_tooltip.blit(title_right, (title_x_right, title_y + 3))

        # Draw horizontal line below title
        line_y = title_y + self.title_right_text.get_height() + 3
        pygame.draw.line(self.cached_tooltip, Const.COLOR_BLACK, (Const.TOOLTIP_PADDING, line_y),
                         (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING, line_y), 4)
    def draw_tooltip_section_headers(self):
        """Draw Forward Pass and Backward Pass headers with optional formula."""
        title_y = Const.TOOLTIP_PADDING
        section_header_y = title_y + self.title_right_text.get_height() + 10

        # Forward Pass
        header_forward = self.font_section.render("Forward Pass", True, Const.COLOR_BLACK)
        self.cached_tooltip.blit(header_forward, (Const.TOOLTIP_PADDING, section_header_y))

        # Backward Pass with formula
        backward_text = " Backward Pass"
        if (hasattr(self.TRI.config, 'optimizer') and hasattr(self.TRI.config.optimizer, 'formula')
                and self.TRI.config.optimizer.formula):
            backward_text += f": {self.TRI.config.optimizer.formula}"

        header_backward = self.font_section.render(backward_text, True, Const.COLOR_BLACK)
        backward_x = self.x_coord_for_col(6)
        self.cached_tooltip.blit(header_backward, (backward_x, section_header_y))

    def draw_tooltip_columns(self):
        """Draw all tooltip columns with proper alignment."""
        x_offset = Const.TOOLTIP_PADDING
        title_offset = self.title_right_text.get_height() + 5

        for col_index, (column, col_width) in enumerate(zip(self.tooltip_columns, self.column_widths)):
            for row_index, text in enumerate(column):
                self.draw_tooltip_cell(text, col_index, row_index, x_offset, col_width, title_offset)
            x_offset += col_width

    def draw_tooltip_cell(self, text, col_index, row_index, x_offset, col_width, title_offset):
        """Draw a single cell in the tooltip."""
        text_color = self.get_text_color(col_index, row_index, text)
        text = self.smart_format_for_popup(text)
        label = self.font_body.render(str(text), True, text_color)
        text_rect = label.get_rect()

        y_pos = Const.TOOLTIP_HEADER_PAD + title_offset + row_index * Const.TOOLTIP_ROW_HEIGHT + Const.TOOLTIP_PADDING

        if self.is_right_aligned(text, row_index):
            text_rect.topright = (x_offset + col_width - Const.TOOLTIP_PADDING, y_pos)
        else:
            text_rect.topleft = (x_offset + Const.TOOLTIP_PADDING, y_pos)

        self.cached_tooltip.blit(label, text_rect)

    def position_and_blit_tooltip(self):
        """Position tooltip near mouse and blit to screen."""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        tooltip_x = self.adjust_position(mouse_x + Const.TOOLTIP_PLACEMENT_X, Const.TOOLTIP_WIDTH, Const.SCREEN_WIDTH)
        tooltip_y = self.adjust_position(mouse_y + Const.TOOLTIP_PLACEMENT_Y, Const.TOOLTIP_HEIGHT, Const.SCREEN_HEIGHT)
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
        self.tooltip_columns.extend(self.tooltip_columns_for_backprop())
        self.set_column_widths()

    # DisplayModel__Neuron_Base.py

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
    def tooltip_columns_for_backprop(self):
        """Compose the final list-of-lists depending on single vs batch mode."""
        is_batch = self.config.batch_size > 1

        # 1) Update block (always present)
        update_cols = self.tooltip_columns_for_backprop_update(is_batch)

        # 2) In batch mode, include the 'joined' finalize block # NEW - always try to show finalizer columns
        batch_finalize_cols = self.tooltip_columns_for_backprop_finalize(is_batch)
        std_finalize_cols = self.tooltip_columns_for_backprop_standard_finale()
        cols = update_cols + batch_finalize_cols + std_finalize_cols
        # 3) Finally inject your error signal section

        return cols
        return self.tooltip_columns_for_error_signal_calculation(cols)

        # DisplayModel__Neuron_Base.py
        # DisplayModel__Neuron_Base.py
        # DisplayModel__Neuron_Base.py

    def build_simple_columns(self, headers, rows):
        """
        Given:
          headers = ['h1', 'h2', 'h3', ...]
          rows    = [(val1, val2, val3, ...), ...]
        Returns list-of-lists:
          [
            ['h1', row1[0], row2[0], ...],
            ['h2', row1[1], row2[1], ...],
            ['h3', row1[2], row2[2], ...],
          ]
        """
        columns = [[h] for h in headers]

        for row in rows:
            for i, val in enumerate(row):
                columns[i].append(val)


        return columns

    def tooltip_columns_for_backprop_update(self, is_batch: bool, num_args: int = 5):

        # Use actual headers captured from the optimizer dict.
        if self.TRI.backprop_headers is not None:
            headers = self.TRI.backprop_headers
            num_args = len(headers)
        else:
            # Defensive fallback (shouldn't happen after first recording)
            headers = [f"h{i + 1}" for i in range(num_args)]


        # Build SELECT clause dynamically
        arg_fields = [f"arg_{i + 1}" for i in range(num_args)]

        sql = f"""
                SELECT {', '.join(arg_fields)}
                  FROM WeightAdjustments
                 WHERE run_id    = ?
                   AND epoch     = ?
                   AND sample_id = ?
                   AND nid       = ?
                 ORDER BY weight_id ASC
            """
        params = (self.run_id,
                  self.model.display_epoch,
                  Const.vcr.CUR_SAMPLE,
                  self.nid)
        rows = Const.dm.db.query(sql, params, as_dict=False)

        return self.build_simple_columns(headers, rows)
    def tooltip_columns_for_backprop_finalize(self, is_batch: bool):
        # Not needed for batch_size=1, return empty
        return []


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

    # DisplayModel__Neuron_Base.py, tooltip_columns_for_error_signal_calculation
    def tooltip_columns_for_error_signal_calculation(self, all_cols):
        """Adds the error signal calculation section to the tooltip."""
        # Add spacing row and divider text
        for i in range(len(all_cols)):
            if i == 0:
                all_cols[0].append("BLAME SOURCES BELOW")
            else:
                all_cols[i].append(" ")

        if self.is_output:
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
        Builds the bottom blame sources section with proper column structure.
        Creates 4 clean columns: From | Weight | Their Blame | Slice
        """
        # Add "BLAME SOURCES BELOW" divider in first column only
        # (other columns already have spacing from previous section)

        # Create 4 new columns for blame section
        col_from = ["From"]
        col_weight = ["Weight"]
        col_their_blame = ["Their Blame"]
        col_slice = ["Slice"]

        # Get data from database
        weights, blames = self.get_elements_of_backproped_error()

        if weights:
            # Generate "From" labels for downstream neurons
            downstream_layer = self.layer + 1
            feeds_to_output = (self.layer == len(self.config.architecture) - 2)

            # Add data rows - one per downstream connection
            for i, (weight, blame) in enumerate(zip(weights, blames)):
                if feeds_to_output:
                    from_label = "Out" if len(weights) == 1 else f"Out-{i}"
                else:
                    from_label = f"{downstream_layer}-{i}"

                slice_value = weight * blame

                col_from.append(from_label)
                col_weight.append(weight)
                col_their_blame.append(blame)
                col_slice.append(slice_value)

            # Calculate totals
            slices = [w * b for w, b in zip(weights, blames)]
            blame_from_all = sum(slices)
            accepted_blame = blame_from_all * self.activation_gradient

            # Add summary section with proper alignment
            col_from.append("")
            col_weight.append("")
            col_their_blame.append("Sum of Slices:")
            col_slice.append(blame_from_all)

            col_from.append("")
            col_weight.append("")
            col_their_blame.append("x Act Gradient:")
            col_slice.append(self.activation_gradient)

            col_from.append("")
            col_weight.append("")
            col_their_blame.append("= Accepted Blame:")
            col_slice.append(accepted_blame)

        # Append the 4 blame columns to existing columns
        all_cols.extend([col_from, col_weight, col_their_blame, col_slice])

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



################COLUMN WIDTHS################COLUMN WIDTHS################COLUMN WIDTHS
    ################COLUMN WIDTHS################COLUMN WIDTHS################COLUMN WIDTHS
    def set_column_widths(self):
        """
        Dynamically calculates column widths and tracks maximums to prevent shrinking.
        """
        # Initialize max width tracking if not exists
        if not hasattr(self, 'max_column_widths'):
            self.max_column_widths = []

        # Local overrides - key is column index, value is fixed width
        column_overrides = {
            # Add overrides here as needed
        }

        self.column_widths = []

        for col_index, column in enumerate(self.tooltip_columns):
            # Check for override first
            if col_index in column_overrides:
                width = column_overrides[col_index]
            elif col_index < 6:
                # Forward pass columns
                width = self.calculate_forward_pass_column_width(column)
            else:
                # Backward pass columns
                width = self.calculate_backward_pass_column_width(column, col_index)

            # Track maximum width ever seen for this column
            if col_index >= len(self.max_column_widths):
                self.max_column_widths.append(width)
            else:
                # Only grow, never shrink
                width = max(width, self.max_column_widths[col_index])
                self.max_column_widths[col_index] = width

            self.column_widths.append(width)

        # Update total tooltip width
        Const.TOOLTIP_WIDTH = sum(self.column_widths) + (Const.TOOLTIP_PADDING * 2)

    def calculate_forward_pass_column_width(self, column):
        """Calculate width for forward pass columns based on header."""
        MIN_WIDTH = 35

        if len(column) > 0:
            value = column[0]  # Use header
            text = self.smart_format_for_popup(value)
            rendered = self.font_body.render(str(text), True, Const.COLOR_BLACK)
            width = rendered.get_width() + (Const.TOOLTIP_PADDING * 2)
        else:
            width = MIN_WIDTH

        return max(width, MIN_WIDTH)

    def calculate_backward_pass_column_width(self, column, col_index):
        """Calculate width for backward pass columns, scanning all values except long labels."""
        MIN_WIDTH = 35
        MAX_LABEL_LENGTH = 15

        # Add extra width for last column
        is_last_column = col_index == len(self.tooltip_columns) - 1
        extra_padding = 20 if is_last_column else 0

        max_width = MIN_WIDTH
        for value in column:
            text = self.smart_format_for_popup(value)
            text_str = str(text)

            # Skip long text labels
            if isinstance(value, str) and len(text_str) > MAX_LABEL_LENGTH:
                continue

            rendered = self.font_body.render(text_str, True, Const.COLOR_BLACK)
            text_width = rendered.get_width() + (Const.TOOLTIP_PADDING * 2) + extra_padding
            max_width = max(max_width, text_width)

        return max(max_width, MIN_WIDTH)


    ####################################################################################
    ########################################################################################

    # DisplayModel__Neuron_Base, draw_blame_sources_section for HIDDEN

    def draw_blame_sources_section(self):
        """Renders blame sources section below main tooltip."""
        if self.is_output:
            self.draw_blame_sources_section_output()
            return


        weights, blames = self.get_elements_of_backproped_error()
        if not weights:            return

        x, y = self.calculate_blame_start_position()
        y = self.draw_section_header(x, y)
        y = self.draw_blame_table(x, y, weights, blames)
        y = self.draw_table_end_line(x, y)
        y = self.draw_blame_summary(x, y, weights, blames)
        self.draw_bridge_text(x, y)  # No return needed, it's the last call

    # DisplayModel__Neuron_Base.py - add this method after draw_blame_sources_section

    def draw_blame_sources_section_output(self):
        """Renders blame source section for OUTPUT neuron - where blame originates."""
        x, y = self.calculate_blame_start_position()

        # Header
        header = self.font_body.render("BLAME SOURCES", True, Const.COLOR_BLACK)
        self.cached_tooltip.blit(header, (x, y + 5))
        y += header.get_height() + 15

        # Explanation text
        lines = [
            "This is where blame ORIGINATES.",
            "No chain rule needed - blame flows",
            "directly from the prediction error.",
            ""
        ]
        for line in lines:
            text = self.font_body.render(line, True, Const.COLOR_BLACK)
            self.cached_tooltip.blit(text, (x, y))
            y += text.get_height() + 2

        y += 10

        # The calculation with contextual hints
        accepted_blame = self.loss_gradient * self.activation_gradient
        calc_lines = [
            ("Loss Gradient:", self.loss_gradient, "(NOT to be confused with loss value which BP ignores.)"),
            ("× Act Gradient:", self.activation_gradient, "(from 'Forward Prop' look up and left)"),
            ("= Accepted Blame:", accepted_blame, "")
        ]

        label_x = x + 20
        value_x = x + 150
        hint_x = x + 220

        for label_text, value, hint in calc_lines:
            label = self.font_body.render(label_text, True, Const.COLOR_BLACK)
            val = self.font_body.render(self.smart_format_for_popup(value), True, Const.COLOR_BLACK)
            self.cached_tooltip.blit(label, (label_x, y))
            self.cached_tooltip.blit(val, (value_x, y))
            if hint:
                hint_surf = self.font_body.render(hint, True, Const.COLOR_GRAY_DARK)
                self.cached_tooltip.blit(hint_surf, (hint_x, y))
            y += Const.TOOLTIP_ROW_HEIGHT

        y += 15

        # Bridge text
        bridge = self.font_section.render("ACCEPTED BLAME IS COLUMN 2 ABOVE", True, Const.COLOR_BLACK)
        self.cached_tooltip.blit(bridge, (x, y))
    def calculate_blame_start_position(self):
        x = Const.TOOLTIP_PADDING + sum(self.column_widths[:6]) +10
        leaky_row_index = len(self.weights_before) + 1  # +1 for "Raw Sum" row
        y = self.y_coord_for_row(leaky_row_index)
        return x, y

    def draw_section_header(self, x, y):
        header = self.font_body.render("BLAME SOURCES", True, Const.COLOR_BLACK)
        self.cached_tooltip.blit(header, (x, y + 5))  # Added +5 to lower it from ceiling
        y += header.get_height() + 10  # Increased from +5 to +10 for more space
        #pygame.draw.line(self.cached_tooltip, Const.COLOR_GRAY_DARK, (x, y),                         (Const.TOOLTIP_WIDTH - Const.TOOLTIP_PADDING, y), 2)
        return y + 10

    def draw_bridge_text(self, x, y):
        bridge = self.font_section.render("ACCEPTED BLAME IS COLUMN 2 ABOVE", True, Const.COLOR_BLACK)
        self.cached_tooltip.blit(bridge, (x, y))
        return y + bridge.get_height() + 10

    def draw_blame_summary(self, x, y, weights, blames):
        slices = [w * b for w, b in zip(weights, blames)]
        blame_from_all = sum(slices)
        accepted_blame = blame_from_all * self.activation_gradient

        widths = [60, 80, 100, 100]
        slice_col_x = x + sum(widths[:3])

        lines = [
            ("Sum of Slices:", blame_from_all),
            ("x Act Gradient:", self.activation_gradient),
            ("= Accepted Blame:", accepted_blame)
        ]

        for label_text, value in lines:
            label = self.font_body.render(label_text, True, Const.COLOR_BLACK)
            val = self.font_body.render(self.smart_format_for_popup(value), True, Const.COLOR_BLACK)
            self.cached_tooltip.blit(label, (slice_col_x - label.get_width() - 10, y))
            val_rect = val.get_rect(topright=(slice_col_x + widths[3] - 5, y))
            self.cached_tooltip.blit(val, val_rect)
            y += Const.TOOLTIP_ROW_HEIGHT

        return y + 5

    def draw_blame_table(self, x, y, weights, blames):
        headers = ["From", "Weight", "Their Blame", "Slice"]
        widths = [60, 80, 100, 100]
        self.draw_table_row(x, y, headers, widths)
        y += Const.TOOLTIP_ROW_HEIGHT
        downstream_layer = self.layer + 1
        feeds_output = (self.layer == len(self.config.architecture) - 2)
        for i, (weight, blame) in enumerate(zip(weights, blames)):
            from_label = "Out" if feeds_output and len(
                weights) == 1 else f"Out-{i}" if feeds_output else f"{downstream_layer}-{i}"
            row = [from_label, weight, blame, weight * blame]
            self.draw_table_row(x, y, row, widths)
            y += Const.TOOLTIP_ROW_HEIGHT
        return y + 10

    def draw_table_row(self, x, y, values, widths):
        font = self.font_body
        curr_x = x
        for i, (val, width) in enumerate(zip(values, widths)):
            text = val if i == 0 else self.smart_format_for_popup(val)
            rendered = font.render(text, True, Const.COLOR_BLACK)
            if i == 0:
                self.cached_tooltip.blit(rendered, (curr_x + 5, y))
            else:
                rect = rendered.get_rect(topright=(curr_x + width - 5, y))
                self.cached_tooltip.blit(rendered, rect)
            curr_x += width

    # DisplayModel__Neuron_Base, draw_table_end_line (new method after draw_blame_table)

    def draw_table_end_line(self, x, y):
        """Draw line at end of table. Returns next y position."""
        pygame.draw.line(self.cached_tooltip, Const.COLOR_GRAY_DARK, (x, y), (x + sum([60, 80, 100, 100]), y), 2)
        return y + 10

    # DisplayModel__Neuron_Base, draw_bridge_text (move to end, after summary)

    def draw_bridge_text(self, x, y):
        bridge = self.font_section.render("ACCEPTED BLAME IS COLUMN 2 ABOVE", True, Const.COLOR_BLACK)
        self.cached_tooltip.blit(bridge, (x, y))