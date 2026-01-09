import pygame
from src.NNA.engine.Utils import *
from src.NNA.utils.general_text import smart_format
from src.NNA.utils.pygame import draw_rect_with_border, get_text_rect, draw_text_with_background, check_label_collision
from src.NeuroForge import Const


class DisplayModel__NeuronWeights:
    """
    DisplayModel__NeuronWeights is created by DisplayModel_Neuron.
    It is an instance of visualizer following the  strategy pattern.
    It holds a reference to DisplayModel__Neuron which is where it gets most of it's information

    This class has the following primary purposes:
    1) Initialize - store information that will not change ( Margins, padding, bar height, etc.
    2) Calculate changing information specific to this visualization (i.e. bar width when weight grows)
    3) Draw the "Standard" components of the neuron.  (Body, Banner, and Banner Text)
    4) Invoke the appropriate "Visualizer" to draw the details of the Neuron
    """

    def __init__(self, neuron, ez_printer):
        # Configuration settings
        self.padding_top                = 3
        self.gap_between_bars           = 0
        self.gap_between_weights        = 2
        self.BANNER_HEIGHT              = 29    # 4 pixels above + 26 pixels total height #TODO this should be tied to drawing banner rather than a const
        self.right_margin               = 40  #  Space reserved for activation visualization
        self.right_margin_pad           = 30 # Allow a bit of space so they are not right on top of each other
        self.padding_bottom             = 3
        self.bar_border_thickness       = 1
        self.min_height_per_label       = 27

        # Neuron attributes
        self.neuron                     = neuron  # ✅ Store reference to parent neuron
        self.num_weights                = 0
        self.bar_height                 = 0
        self.max_activation             = 0
        #print(f"neuron height={neuron.location_height}")
        # Weight mechanics
        self.ez_printer                 = ez_printer
        self.my_fcking_labels           = [] # WARNING: Do NOT rename. Debugging hell from Python interpreter defects led to this.
        self.need_label_coord           = True #track if we recorded the label positions for the arrows to point from

        self.num_weights                = len(self.neuron.weights)
        self.neuron_height              = self.neuron.location_height
        self.print_weight_lbl           = False

        if self.num_weights > 0:
            self.bar_height                 = self.calculate_bar_height(num_weights=self.num_weights, neuron_height=self.neuron_height, padding_top=self.padding_top,padding_bottom=self.padding_bottom, gap_between_bars= self.gap_between_bars,gap_between_weights=self.gap_between_weights)
            height_per_label                = self.neuron_height/self.num_weights
            #if self.neuron.nid==6:          print(f"height_per_label={height_per_label}")
            self.print_weight_lbl           = height_per_label > self.min_height_per_label
            #ez_debug(neuron=self.neuron.nid, bar=self.bar_height, label=height_per_label)

    def recalculate_layout(self):
        """Recalculate bar positioning when neuron is resized."""
        # Update stored neuron height
        self.neuron_height = self.neuron.location_height

        # Clear old label positions so they get recalculated
        self.my_fcking_labels = []
        self.need_label_coord = True

        # Recalculate bar height based on new neuron height
        if self.num_weights > 0:
            self.bar_height = self.calculate_bar_height(
                num_weights=self.num_weights,
                neuron_height=self.neuron_height,
                padding_top=self.padding_top,
                padding_bottom=self.padding_bottom,
                gap_between_bars=self.gap_between_bars,
                gap_between_weights=self.gap_between_weights
            )

            # Recalculate whether to print weight labels
            height_per_label = self.neuron_height / self.num_weights
            self.print_weight_lbl = height_per_label > self.min_height_per_label

    def render(self):
        self.draw_weight_bars()
        self.draw_blame_bar_in_header()
        self.draw_computation_line()
        self.draw_activation_bar()

    def calculate_bar_height(self, num_weights, neuron_height, padding_top, padding_bottom, gap_between_bars, gap_between_weights):
        """
        Calculate the height of each weight bar dynamically based on available space.

        :param num_weights: Number of weights for the neuron
        :param neuron_height: Total height of the neuron
        :param padding_top: Space above the first set of bars
        :param padding_bottom: Space below the last set of bars
        :param gap_between_bars: Gap between the two bars of the same weight
        :param gap_between_weights: Gap between different weights
        :return: The calculated height for each individual bar
        """
        # Calculate available space after removing padding
        available_height = neuron_height - (padding_top + padding_bottom + self.BANNER_HEIGHT)

        # Each weight has two bars, so total bar slots = num_weights * 2
        total_gaps = (num_weights * gap_between_weights) + (num_weights * 2 - 1) * gap_between_bars

        # Ensure the remaining space is distributed across all bars
        if total_gaps >= available_height:
            return 1
        #    raise ValueError(f"Not enough space in neuron height to accommodate weights and gaps.\nNeuron Height: {neuron_height}, Available Height: {available_height},\nTotal Gaps: {total_gaps}, Computed Bar Height: {bar_height}" )

        # Compute actual bar height
        total_bar_height = available_height - total_gaps
        bar_height = total_bar_height / (num_weights * 2)

        return bar_height



    def will_computation_line_fit(self, left_text, right_text, font, available_width):
        """
        Checks if both left and right text will fit without overlapping.
        Returns True if they fit, False if we need to fallback.
        """
        # Render to measure widths
        left_width = font.size(left_text)[0]
        right_width = font.size(right_text)[0]

        # Add padding (5px left margin + 6px right margin + 10px gap between texts)
        total_width = left_width + right_width + 12

        return total_width <= available_width

    def draw_computation_line(self):
        """
        Draws the computation line at bottom showing: function(weighted_sum) = output
        Falls back to just showing output if the full format doesn't fit.
        """
        if self.neuron.am_really_short:
            return

        weighted_sum = self.neuron.raw_sum
        activation_fn = self.neuron.activation_function if self.neuron.activation_function else "f"
        output_value = self.neuron.activation_value

        # Draw background rect for entire computation line
        comp_line_height = 20
        comp_line_y = self.neuron.location_top + self.neuron.location_height
        comp_rect = pygame.Rect(
            self.neuron.location_left + 2,
            comp_line_y - 4,
            self.neuron.location_width - 4,
            comp_line_height
        )
        pygame.draw.rect(self.neuron.screen, Const.COLOR_COMP_LINE_BG, comp_rect)
        pygame.draw.rect(self.neuron.screen, Const.COLOR_BLUE_MIDNIGHT, comp_rect, 12)

        # Prepare text and font
        left_text = f"{activation_fn}({smart_format(weighted_sum)}) ="
        right_text = smart_format(output_value)
        font = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT + 2)
        available_width = self.neuron.location_width - 4

        # Check if full format fits
        if self.will_computation_line_fit(left_text, right_text, font, available_width):
            # Draw left-aligned function part
            text_surface = font.render(left_text, True, Const.COLOR_WHITE)
            self.neuron.screen.blit(text_surface, (self.neuron.location_left + 5, comp_line_y - 3))

            # Draw right-aligned output value
            text_surface = font.render(right_text, True, Const.COLOR_WHITE)
            text_width = text_surface.get_width()
            right_x = self.neuron.location_left + self.neuron.location_width - text_width - 6
            self.neuron.screen.blit(text_surface, (right_x, comp_line_y - 3))
        else:
            # Fallback: just draw the activation value, centered
            text_surface = font.render(right_text, True, Const.COLOR_WHITE)
            text_width = text_surface.get_width()
            center_x = self.neuron.location_left + (self.neuron.location_width - text_width) // 2
            self.neuron.screen.blit(text_surface, (center_x, comp_line_y - 3))



        def get_max_accepted_blame_for_iteration(self):
            """
            Get max absolute accepted_blame for CURRENT sample (sample mode)
            or max of epoch-averaged blames (epoch mode).
            """
            run_id = Const.TRIs[0].run_id

            if Const.vcr.blame_mode == "epoch":
                SQL = """
                    SELECT MAX(avg_abs_blame) AS max_accepted_blame
                    FROM (
                        SELECT AVG(ABS(accepted_blame)) AS avg_abs_blame
                        FROM Neuron 
                        WHERE run_id = ? AND epoch = ?
                        GROUP BY nid
                    )
                """
                result = self.db.query(SQL, (run_id, Const.vcr.CUR_EPOCH))
            else:
                SQL = """
                    SELECT MAX(ABS(accepted_blame)) AS max_accepted_blame
                    FROM Neuron WHERE run_id = ? AND epoch = ? AND sample = ? 
                """
                result = self.db.query(SQL, (run_id, Const.vcr.CUR_EPOCH, Const.vcr.CUR_SAMPLE))

            self.max_blame = result[0]['max_accepted_blame'] if result and result[0]['max_accepted_blame'] is not None else 1.0
            if not hasattr(self, 'historical_max_blame'):
                self.historical_max_blame = self.max_blame
            else:
                self.historical_max_blame = max(self.historical_max_blame, self.max_blame)



    def draw_blame_bar_in_header(self):
        """
        Draws the blame bar in the header area, growing RIGHT-TO-LEFT.
        Mode toggles between sample-specific blame and epoch-averaged blame.
        """
        if Const.dm.max_blame == 'Unknown':  # Safety check
            return

        neuron_x = self.neuron.location_left + self.neuron.location_width  # Start at right edge
        neuron_y = self.neuron.location_top + 4  # Top of neuron (match banner position)

        # Determine which blame value to use based on mode
        if Const.vcr.blame_mode == "epoch":
            blame_value = self.neuron.avg_err_sig_for_epoch  # Always positive (mean of absolutes)
            bar_color = Const.COLOR_BLUE_SKY  # Blue for epoch mode
            label_text = f"Epoch Avg:"
            label_text = f"Epoch Avg:{smart_format(blame_value)}"
            #label_text = f"Epoch Avg"
        else:  # "sample" mode
            blame_value = self.neuron.blame  # Can be positive or negative
            bar_color = Const.COLOR_FOR_ACT_POSITIVE if blame_value >= 0 else Const.COLOR_FOR_ACT_NEGATIVE
            label_text = f"Sample {smart_format(blame_value)}"

        # ADAPTIVE SCALING: Shrink max bar width as network quiets down

        # Reserve space for banner text to prevent overlap
        reserved_for_text = self.neuron.banner_text_width + 10  # Text width + small gap
        base_max_width = self.neuron.location_width - reserved_for_text

        max_bar_width = self.neuron.location_width - reserved_for_text - 10
        # Calculate intensity ratio (how active is current epoch vs historical max)
        if hasattr(Const.dm, 'historical_max_blame') and Const.dm.historical_max_blame > 0:
            intensity_ratio = Const.dm.max_blame / Const.dm.historical_max_blame
            # Scale from 40 pixels (quiet) to base_max_width (active)
            max_bar_width = 40 + (base_max_width - 40) * min(1.0, intensity_ratio)
        else:
            max_bar_width = base_max_width
        #ez_debug(max_bar_width=max_bar_width)
        # Calculate blame bar length (grows left from right edge)
        blame_magnitude = abs(blame_value)
        if Const.dm.max_blame==0: Const.dm.max_blame=.0000001
        #max_bar_width = self.neuron.location_width - 80  # Leave space for neuron ID on left
        bar_width = (blame_magnitude / Const.dm.max_blame) * max_bar_width
        if bar_width > max_bar_width:
            bar_width = max_bar_width

        # Define bar position (grows LEFT from right edge)
        bar_rect = pygame.Rect(
            neuron_x - bar_width - 5,  # Start from right edge, grow left
            neuron_y + 5,  # Vertically centered in banner
            bar_width,
            self.BANNER_HEIGHT - 10  # Height fits in banner
        )

        # Draw the blame bar
        draw_rect_with_border(self.neuron.screen, bar_rect, bar_color, 2)

        # Determine what text fits
        text_to_show = self.get_text_that_fits(
            full_text=label_text,
            fallback_text=smart_format(blame_value),
            available_width=bar_width,
            font_size=18
        )
        #text_to_show=label_text #Temp to put values always

        if text_to_show:
            font = pygame.font.Font(None, 18)
            text_surface = font.render(text_to_show, True, Const.COLOR_BLACK)
            text_rect = text_surface.get_rect(midright=(
                neuron_x - 10,  # Right edge of bar with padding
                neuron_y + self.BANNER_HEIGHT // 2  # Vertically centered
            ))
            self.neuron.screen.blit(text_surface, text_rect)

    def get_text_that_fits(self, full_text, fallback_text, available_width, font_size, padding=10):
        """
        Determines which text (if any) will fit in the available space.

        Tries in order:
        1. full_text (e.g., "Epoch Avg 0.42")
        2. fallback_text (e.g., "0.42")
        3. None (if even the number doesn't fit)

        Args:
            full_text: Complete descriptive text
            fallback_text: Shortened fallback (just the number)
            available_width: Width in pixels available for text
            font_size: Font size for measurement
            padding: Safety margin (default 10px for breathing room)

        Returns:
            str or None: Text that fits, or None if nothing fits
        """
        font = pygame.font.Font(None, font_size)

        # Try full text first
        full_width = font.size(full_text)[0]
        if full_width + padding <= available_width:
            return full_text

        # Try fallback text
        fallback_width = font.size(fallback_text)[0]
        if fallback_width + padding <= available_width:
            return fallback_text

        # Nothing fits
        return None

    def draw_weight_bars(self):
        """
        Draw all weight bars inside the neuron, considering padding, spacing, and bar height.
        This function ensures bars are evenly spaced and positioned inside the neuron.
        """

        # Compute the starting X and Y positions as well as length of bars
        start_x     = self.neuron.location_left + 5  # Small left padding for visibility
        start_y     = self.neuron.location_top + self.padding_top + self.BANNER_HEIGHT # Start from the top padding
        bar_lengths = self.calculate_weight_bar_lengths()   # For example magnitude of Weight

        for i, (bar_self, bar_global) in enumerate(bar_lengths):    # Compute vertical position of this weight's bars
            y_pos = start_y + i * (self.bar_height * 2 + self.gap_between_weights)

            # Call function to draw the two bars for this weight
            self.draw_two_bars_for_one_weight(start_x, y_pos, bar_self, bar_global, self.bar_height, self.gap_between_bars, self.neuron.weights_before[i],i)

        if len(self.my_fcking_labels) > 0:
            self.need_label_coord = False #Only need to record on first pass.

    def get_max_bar_width(self):
        """
        Calculates the absolute maximum width a weight bar can have.
        This ensures bars never draw past the neuron boundary.
        """
        neuron_width = self.neuron.location_width - self.right_margin * .5
        if neuron_width < 20:
            neuron_width = 20

        max_width = neuron_width - self.right_margin_pad
        return max_width


    def calculate_weight_bar_lengths(self):
        """
        Calculates the bar lengths for visualizing weight magnitudes.
        - The first bar represents the weight's magnitude relative to itself (normalized per weight).
        - The second bar represents the weight's magnitude relative to all weights (global normalization).
        - Bars are scaled relative to the neuron's width minus the right margin.
        """

        # Adjust neuron width to account for the right margin
        neuron_width = self.neuron.location_width - self.right_margin * .5
        if neuron_width < 20:  # Prevent bars from being too small
            neuron_width = 20

        bar_lengths = []
        #if self.neuron.nid ==0:            ez_debug(wts=self.neuron.weights, wts_before=self.neuron.weights_before)
        for i, weight in enumerate(self.neuron.weights_before):
            abs_weight = abs(weight)  # Use absolute value for visualization

            # Normalize relative to this weight's historical max
            self_max = self.neuron.max_per_weight[i] if self.neuron.max_per_weight[i] != 0 else 1
            norm_self = abs_weight / self_max  # Scale between 0 and 1

            # Normalize relative to the absolute global max weight   # Scale between 0 and 1
            norm_global = abs_weight / (Const.MAX_WEIGHT if Const.MAX_WEIGHT != 0 else 1)

            # Scale to neuron width (so bars fit inside the neuron)
            bar_self = norm_self * (neuron_width - self.right_margin_pad)
            bar_global = norm_global * (neuron_width - self.right_margin_pad)

            # Apply hard limit to prevent bars from drawing past neuron
            max_width = self.get_max_bar_width()
            bar_self = min(bar_self, max_width)
            bar_global = min(bar_global, max_width)

            bar_lengths.append((bar_self, bar_global))
        return bar_lengths

    def draw_two_bars_for_one_weight(self, x, y, width_self, width_global, bar_height, bar_gap, weight_value, weight_id):
        """
        Draws two horizontal bars for a single weight visualization with labels.

        - Top bar = Global max reference.
        - Bottom Bar = Self max reference.
        - Labels are drawn inside if space allows, or outside if bars are too small.
        """

        # Create rectangles first
        global_rect = pygame.Rect(x, y, width_global, bar_height)  # Orange bar
        self_rect = pygame.Rect(x, y + bar_height + bar_gap, width_global, bar_height)  # Green bar
        #self_rect = pygame.Rect(x, y + bar_height + bar_gap, width_self, bar_height)  # Green bar

        color1 = Const.COLOR_FOR_BAR1_POSITIVE if weight_value >= 0 else Const.COLOR_FOR_BAR1_NEGATIVE
        color2 = Const.COLOR_FOR_BAR2_POSITIVE if weight_value >= 0 else Const.COLOR_FOR_BAR2_NEGATIVE

        # Draw bars with borders
        #draw_rect_with_border(self.neuron.screen, global_rect, color1, self.bar_border_thickness)  # Orange (global max)
        #draw_rect_with_border(self.neuron.screen, self_rect, color2, self.bar_border_thickness)  # Green (self max)
        # Draw bars - with borders only if tall enough, otherwise just filled rects
        if bar_height >= Const.WEIGHT_MIN_HEIGHT_OUTLINE:
            draw_rect_with_border(self.neuron.screen, global_rect, color1, self.bar_border_thickness)
            draw_rect_with_border(self.neuron.screen, self_rect, color2, self.bar_border_thickness)
        else:
            # For very thin bars, draw without border so color is visible
            pygame.draw.rect(self.neuron.screen, color1, global_rect)
            pygame.draw.rect(self.neuron.screen, color2, self_rect)



        # Draw labels dynamically based on available space
        label_rects=[]
        label_space = self.draw_weight_label(weight_value, global_rect, bar_height)  # Label for global bar
        # THIS IS READY TO GO ======-> label_space = self.draw_weight_label(weight_value, self_rect)  # Label for self bar
        label_rects.append(label_space)
        label_rects.append(label_space)
        self.draw_weight_index_label(weight_id, y+self.bar_height-9, label_rects)

    def draw_weight_label(self, value_to_print, rect, bar_height):
        """
        Draws a weight label with a background for readability.

        - If the bar is wide enough, places the label inside the bar.
        - If the bar is too small, places the label outside (to the right).
        - Uses a black semi-transparent background to improve contrast.
        """

        # Define the minimum width required to place the text inside the bar
        min_label_width = 30

        # Create font and render text
        font = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)
        text_surface = font.render(smart_format(value_to_print), True, Const.COLOR_WHITE)  # White text
        text_rect = text_surface.get_rect()
        #ez_debug(bar_height=bar_height)
        # Determine label placement: inside if enough space, otherwise outside
        if rect.width >= min_label_width:
            text_rect.midleft = (rect.left + 5, rect.centery+(bar_height/2))
        else:
            text_rect.midleft = (rect.right + 5, rect.centery+(bar_height/2))  # Place outside to the right

        # Ensure label doesn't go out of bounds
        if text_rect.right > self.neuron.screen.get_width():
            text_rect.right = self.neuron.screen.get_width() - 5

        if not self.print_weight_lbl: return text_rect
        # Draw a semi-transparent background behind the text for readability
        bg_rect = text_rect.inflate(4, 2)  # Slight padding
        pygame.draw.rect(self.neuron.screen, Const.COLOR_LABEL_BG_TRANSPARENT, bg_rect)  # Dark transparent background

        # Render text onto screen
        self.neuron.screen.blit(text_surface, text_rect)
        return text_rect # to make sure weight index label doesn't collide.

    def draw_weight_index_label(self, weight_index, y_pos,existing_labels_rects):
        """
        Draws a small label with the weight index on the left wall of the neuron,
        positioned in the middle between the two bars.

        :param weight_index: The index of the weight.
        :param y_pos: The y-position of the weight bars.
        :param existing_labels_rects: list of rects for other labels that might collide.
        """
        # Compute label position
        label_x = self.neuron.location_left  + 5 # Slightly left of the neuron
        label_y = y_pos   # Middle of the two bars

        # Format the label text and get text rect
        label_text = f"Wt #{weight_index}"
        if weight_index == 0:
            label_text = "Bias"
        text_rect = get_text_rect(label_text, Const.FONT_SIZE_WEIGHT) #Get rect for index label.
        text_rect.topleft = label_x, label_y

        if self.neuron.layer == 0 and self.neuron.location_left > text_rect.width + 5:
            label_x = self.neuron.location_left - text_rect.width -  3
            draw_text_with_background(self.neuron.screen, label_text, label_x, label_y, Const.FONT_SIZE_WEIGHT, Const.COLOR_WHITE, Const.COLOR_BLUE, border_color=Const.COLOR_BLACK)

            # Record label loc for input arrows to go to.
            if self.need_label_coord:
                self.my_fcking_labels.append((label_x-text_rect.width * 0.2, label_y))
            return
        if self.need_label_coord:
            self.my_fcking_labels.append((label_x,label_y))

        # Check if there is a collision
        if not check_label_collision(text_rect, existing_labels_rects):
            draw_text_with_background(self.neuron.screen, label_text, label_x, label_y, Const.FONT_SIZE_WEIGHT, Const.COLOR_WHITE, Const.COLOR_BLUE, border_color=Const.COLOR_BLACK)

    def draw_activation_bar(self):
        """
        Draws the activation bar inside the right margin of the neuron.
        - Bar height is scaled relative to the **global max activation**.
        - The bar is drawn from the **bottom up** with a fixed anchor point.
        - Uses `self.right_margin` as the width.
        """
        bar_rect = self.calculate_activation_bar_rect()
        if bar_rect is None:
            return

        # Choose color based on activation sign
        bar_color = (Const.COLOR_FOR_ACT_POSITIVE if self.neuron.activation_value >= 0
                     else Const.COLOR_FOR_ACT_NEGATIVE)

        # Draw the activation bar
        draw_rect_with_border(self.neuron.screen, bar_rect, bar_color, 2)

    def calculate_activation_bar_rect(self):
        """
        Calculates the activation bar rectangle with a fixed bottom anchor.
        Returns a pygame.Rect positioned to grow upward from a constant bottom position.

        Returns None if max_activation is 0.
        """
        if self.neuron.max_activation == 0:
            return None

        # Define fixed anchor point (bottom of bar)
        neuron_x = self.neuron.location_left + self.neuron.location_width
        neuron_y = self.neuron.location_top
        bottom_y = neuron_y + self.neuron.location_height - 4

        # Calculate bar height as integer (no float rounding issues)
        activation_magnitude = abs(self.neuron.activation_value)
        available_height = self.neuron.location_height - self.BANNER_HEIGHT
        bar_height = int((activation_magnitude / self.neuron.max_activation) * available_height)

        # Cap at maximum
        max_bar_height = self.neuron_height - self.BANNER_HEIGHT - 5
        if bar_height > max_bar_height:
            bar_height = max_bar_height

        # Calculate top position from anchored bottom
        top_y = bottom_y - bar_height

        return pygame.Rect(
            neuron_x - self.right_margin,
            top_y,
            self.right_margin,
            bar_height
        )
