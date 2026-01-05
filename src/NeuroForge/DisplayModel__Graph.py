import pygame

from src.NNA.utils.pygame import draw_gradient_rect
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt





class DisplayModel__Graph():
    #def __init__(self, width_pct=98, height_pct=4.369, left_pct=1, top_pct=0):
    def __init__(self, width, height, left, top, model_surface, run_id, model):
        """Creates a Graph showing MAE over epoch"""
        #super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)
        #super().__init__(width_pct=0, left_pct=0, top_pct=0, height_pct=0,                         pixel_adjust_width=width, pixel_adjust_left=left,                         pixel_adjust_top=top, pixel_adjust_height=height)
        self.model               = model #reference to DisplayManager__Model object
        self.model_surface          = model_surface

        # Positioning
        self.location_left          = left
        self.location_top           = top
        self.location_width         = width
        self.location_height        = height  #129.75
        #ez_debug(height=self.location_height)
        self._plot_scale_factor             =    1.07969    #increases X and Y (down and left)
        self._plot_scale_factor2                = .8869      #increase x relative to y

        SQL = "SELECT epoch, mae as mean_absolute_error FROM RecordEpoch WHERE run_id = ? ORDER BY epoch"
        self.results = Const.dm.db.query(SQL, (run_id,))

        # Build and store the plot surface once.
        self.plot_surface = self.create_plot_surface_from_results()

        # Create header text
        self.header_text = "Error History"
        #self.error_text = f"{smart_format(model.config.lowest_error)} at {model.config.lowest_error_epoch} "
        #self.error_text = "Coming soon"
        self.error_text = ""
        self.font = pygame.font.Font(None, 30)

    def process_an_event(self, event):
        """Handles UI events and sends commands to VCR.
        Also ensures pygame_gui receives events.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            self.check_click(self.model.left,self.model.top)

    def check_click(self, model_x, model_y):
        """
        Check if the mouse is over this neuron.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        neuron_x = model_x + self.location_left
        neuron_y = model_y + self.location_top
        if (neuron_x <= mouse_x <= neuron_x + self.location_width) and (neuron_y <= mouse_y <= neuron_y + self.location_height):
            #print (f"You click me at {mouse_x - self.model.left - self.location_left } of {self.location_width}")
            click_ratio = (mouse_x - self.model.left - self.location_left) / self.location_width
            click_ratio = max(0, min(click_ratio, 1))  # Clamp to [0, 1]
            target_epoch = int(click_ratio * self.model.last_epoch)
            Const.vcr.jump_to_epoch(str(target_epoch))


    def create_plot_surface_from_results(self):
        # Prepare the data
        epochs = [r['epoch'] for r in self.results]
        mae_values = [r['mean_absolute_error'] for r in self.results]

        dpi = 100
        #figsize = (self.location_width  * self._plot_scale_factor / dpi, self.location_height * self._plot_scale_factor *  self._plot_scale_factor2 / dpi)
        figsize = (self.location_width * self._plot_scale_factor / dpi,
                   self.location_height * self._plot_scale_factor2 / dpi)

        # 🔹 Create the figure and axis
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("none")  # Transparent figure background

        ax = fig.add_subplot(111)
        ax.set_facecolor("none")  # Transparent plot background
        # Scale properly in Y dimension
        #max_error = max(mae_values)


        # 🔹 Plot the error line
        ax.plot(epochs, mae_values, marker='o', linestyle='-', color=(1,0,1))
        # Force consistent vertical scaling
        max_error = max(mae_values)
        min_error = min(mae_values)

        if max_error == min_error:
            # Flat line, pad artificially
            max_error += 1e-3
        # SAME AXIS FOR ALL ax.set_ylim(0, Const.GRAPH_FIXED_MAX_ERROR)  # Define a fixed height if helpful
        ax.set_ylim(min_error - 0.05 * abs(max_error), max_error + 0.05 * abs(max_error))
        ax.set_ylim(0, max_error * 1.05)  # ⬅️ slight buffer above the highest point
        #ax.set_ylim(0, Const.GRAPH_FIXED_MAX_ERROR)  # Define a fixed height if helpful


        fig.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


        # 🔹 Remove padding, ticks, labels, and borders
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_xlabel("")
        #ax.set_ylabel("")
        ax.set_xlabel("Epoch", fontsize=14, labelpad=2)
        ax.set_ylabel("Error", fontsize=14, labelpad=2)
        ax.tick_params(axis='x', labelsize=12, length=3)
        ax.tick_params(axis='y', labelsize=12, length=3)

        for spine in ax.spines.values():
            spine.set_visible(False)
        #plt.subplots_adjust(left=0, right=1, top=.97, bottom=0)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.22)


        # 🔹 Force y-axis to start at 0
        #ax.set_ylim(bottom=0)

        # 🔹 Render to ARGB and convert to RGBA for Pygame
        canvas = FigureCanvas(fig)
        canvas.draw()
        raw_data = canvas.tostring_argb()
        width, height = canvas.get_width_height()

        # 🔁 Convert ARGB to RGBA for Pygame
        import numpy as np
        argb_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))
        rgba_array = argb_array[:, :, [1, 2, 3, 0]]  # Reorder channels

        # 🔹 Create Pygame surface
        plot_surface = pygame.image.frombuffer(rgba_array.tobytes(), (width, height), "RGBA")

        plt.close(fig)
        return plot_surface

    def render(self):
        """Called once per pygame loop"""
        # Blit the pre-rendered plot onto the EZSurface's surface.
        #self.model_surface.blit(self.plot_surface, (self.location_left, self.location_top + 30))
        blit_x = self.location_left + (self.location_width * (1 - self._plot_scale_factor))
        blit_y = self.location_top + 30  # Top stays fixed
        self.model_surface.blit(self.plot_surface, (blit_x, blit_y))
        #Create header

        self.draw_graph_frame()

    def update_me(self):
        """Graph will not change"""
        pass

    def draw_graph_frame(self):
        """Draws a frame styled like a neuron with a banner and border."""


        # === Banner ===
        banner_height = self.font.get_height() + 8
        banner_rect = pygame.Rect(self.location_left, self.location_top+ 3, self.location_width, banner_height)
        # === Border (below banner) ===
        body_top = self.location_top + banner_height
        body_height = self.location_height - banner_height
        pygame.draw.rect(
            self.model_surface,
            Const.COLOR_FOR_NEURON_BODY,
            (self.location_left, body_top, self.location_width, body_height),
            border_radius=6,
            width=7
        )
        draw_gradient_rect(self.model_surface, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)
        label_surface = self.font.render(self.header_text, True, Const.COLOR_FOR_NEURON_TEXT)
        self.model_surface.blit(
            label_surface,
            (self.location_left + 5, self.location_top + 5 + (banner_height - label_surface.get_height()) // 2)
        )
        label_surface = self.font.render(self.error_text, True, Const.COLOR_BLACK)
        self.model_surface.blit(
            label_surface,
            (self.location_left + 15, self.location_top + self.location_height - label_surface.get_height() - 5 )
        )