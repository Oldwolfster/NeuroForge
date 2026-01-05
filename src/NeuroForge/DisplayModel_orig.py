import pygame

from src.NNA.utils.general_text import beautify_text
from src.NeuroForge import Const
from src.NeuroForge.ButtonBase import Button_Base
from src.NeuroForge.DisplayArrow import DisplayArrow
from src.NeuroForge.DisplayModel__Graph import DisplayModel__Graph

from src.NeuroForge.DisplayModel__NeuronScaler import DisplayModel__NeuronScaler
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.GeneratorNeuron import GeneratorNeuron
from src.NeuroForge.PopupArchitecture import ArchitecturePopup
from src.NNA.engine import TrainingRunInfo
from src.NNA.engine.Config import Config
from src.NNA.engine.Neuron import Neuron
from src.NeuroForge.DisplayModel__Layer import DisplayModel__Layer
from src_revert.NNA.utils.general_text import smart_format


class DisplayModel(EZSurface):
    __slots__ = ("needs_arrow_rebuild", "layer_control_rects","layers", "TRI", "thresholder", "last_epoch", "input_scaler_neuron", "prediction_scaler_neuron", "layer_width", "hoverlings", "arch_popup","buttons", "config", "neurons", "threshold", "arrows_forward", "run_id", "graph_holder", "graph", "max_blame")
    def __init__(self, TRI: TrainingRunInfo, position: dict )   :
        """Initialize a display model using pixel-based positioning."""
        super().__init__(
            width_pct=0, height_pct=0, left_pct=0, top_pct=0,  # Ignore percent-based positioning
            pixel_adjust_width  = position["width"],
            pixel_adjust_height = position["height"],
            pixel_adjust_left   = position["left"],
            pixel_adjust_top    = position["top"],
            bg_color            = Const.COLOR_FOR_BACKGROUND
        )
        self.hoverlings: list   = []  # ✅ Store the neuron and objects that react when being hovered over
        self.buttons        = []
        self.TRI            = TRI
        self.config         = TRI.config
        self.arch_popup     = ArchitecturePopup(self, self.config)
        self.layer_width        = 0 # Set in GenerateNeuron static class

        self.last_epoch     = TRI.last_epoch
        #print(f"self.last_epoch = {self.last_epoch}")
        self.graph          = None
        self.thresholder    = None
        self.input_scaler_neuron = None
        self.prediction_scaler_neuron = None
        #self.run_id       = TRI.gladiator
        self.run_id       = TRI.run_id
        #self.neurons        = [[] for _ in range(len(self.config.architecture))]  # Nested list by layers
        self.neurons        = []
        self.layers         = []  # Add this new line
        self.arrows_forward = []  # List of neuron connections
        #_, _,self.threshold = config.training_data.get_binary_decision_settings(config.loss_function)

        btn = Button_Base(
                text=self.get_model_button_text(),
                width_pct=10, height_pct=4, left_pct=1, top_pct=1,
                on_click=self.show_info,
                on_hover=lambda: self.arch_popup.show_me(),
                shadow_offset=-5, auto_size=True, my_surface=self.surface,
                text_line2=f"Accuracy: {self.format_percent(TRI.best_accuracy)} ",
                surface_offset=(self.left, self.top))

        self.buttons.append(btn)
        self.hoverlings.extend(self.buttons)

    def get_model_button_text(self):
        """
        Determines the button text for this model.
        Only adds run_id number if multiple models share the same optimizer name.
        """
        optimizer_counts = {}
        for tri in Const.TRIs:
            name = tri.gladiator
            optimizer_counts[name] = optimizer_counts.get(name, 0) + 1

        # Only add number if there are multiple models with this optimizer name
        needs_number = optimizer_counts[self.TRI.gladiator] > 1
        return f"{beautify_text(self.TRI.gladiator)} {self.TRI.run_id}" if needs_number else beautify_text(
            self.TRI.gladiator)


    @property
    def lowest_error(self):
        return f"{smart_format(self.config.lowest_error)} at {self.config.lowest_error_epoch}"

    def format_percent(self, x: float, decimals: int = 2, max_value: float = 1000.0) -> str:
        """
        Format a fraction x (e.g. 0.9999) as a percentage string:
          • two decimal places normally → "99.99%"
          • no trailing .00 → "100%"
          • gradient explosions → "OVERFLOW"
        """
        # Check for gradient explosion (anything beyond max_value)
        if abs(x) > max_value:
            return "OVERFLOW" if x > 0 else "-OVERFLOW"

        # Normal formatting
        s = f"{x:.{decimals}f}"
        s = s.rstrip("0").rstrip(".")
        return s + "%"


    @property
    def display_epoch(self):
        """
        Returns the appropriate epoch to display based on the global VCR state.
        If the model converged early, it freezes at its last recorded epoch.
        """
        #print (f" self.last_epoch={ self.last_epoch} and Const.vcr.CUR_EPOCH_MASTER={Const.vcr.CUR_EPOCH_MASTER}")
        if self.last_epoch is None:
            return Const.vcr.CUR_EPOCH_MASTER
        return min(Const.vcr.CUR_EPOCH_MASTER, self.last_epoch)

    @property
    def display_iteration(self):
        """
        Same logic as display_epoch but for iteration display.
        You could optionally track last_iteration if you want extra granularity.
        """
        return min(Const.vcr.CUR_ITERATION, self.last_iteration or Const.vcr.CUR_ITERATION)

    def update_last_epoch(self, epoch):
        self.last_epoch = epoch

    def update_last_iteration(self, iteration):
        self.last_iteration = iteration

    def initialize_with_model_info(self):
        """Create neurons and connections based on architecture."""
        max_activation = self.get_max_activation_for_model(self.run_id)
        max_blame = self.get_max_blame_for_model(self.run_id)
        GeneratorNeuron.create_neurons(self, max_activation)
        for layer in self.neurons:
            for neuron in layer:
                neuron.model = self

        # Create error over epochs graph
        self.graph = self.create_graph(self.graph_holder)# Add Graph  # MAE over epoch
        Const.dm.eventors.append(self.graph)
        self.render()   #Run once so everything is created
        self.create_neuron_to_neuron_arrows(True)  # Forward pass arrows
        self.create_output_to_thresholder()

        #ez_debug(BD=self.TRI.training_data.is_binary_decision)
        # Create thresholder if problem type is BD
        if  1== 2 : #self.TRI.training_data.is_binary_decision:
            predictor = self.prediction_scaler_neuron
            ez_debug(predictor=predictor)
            self.thresholder = DisplayModel__NeuronScaler(
                self,
                left=predictor.location_left + 10,
                top=predictor.location_top + 10,
                width=predictor.location_width,
                height=predictor.location_height,
                nid=-1,         # Not in data flow
                layer=-1,
                position=0,
                output_layer=0,
                text_version=predictor.text_version,
                run_id=self.TRI.run_id,
                screen=self.surface,
                max_activation=0
            )

    def create_graph(self, gh):
        doublewide= gh.location_width * 2 + 20
        return DisplayModel__Graph(left=gh.location_left, width=doublewide , top=gh.location_top, height=gh.location_height, model_surface=self.surface, run_id=self.run_id, model=self)

    def get_all_layer_buttons(self):
        """Return all button objects from all layers for registration with Display_Manager"""
        all_buttons = []
        for layer in self.layers:
            if hasattr(layer, 'control_buttons'):
                all_buttons.extend(layer.control_buttons)
        return all_buttons



    def render(self):
        """Draw neurons and connections."""
        self.clear()
        self.draw_border()
        #self.draw_layer_controls()
        #self.draw_layer_indicators()
        if self.graph is not None:
            self.graph.render()
#        for connection in self.connections:
#            connection.draw_connection(self.surface)
        if self.input_scaler_neuron is not None:
            self.input_scaler_neuron.draw_neuron()
        if self.prediction_scaler_neuron is not None:
            self.prediction_scaler_neuron.draw_neuron()
        if self.thresholder is not None:
            #print("drawing thresholder")
            self.thresholder.draw_neuron()

        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron()
        # Draw layer controls
        #self.draw_layer_controls()

        for arrow in self.arrows_forward:
            arrow.draw()

        for button in self.buttons:
            button.draw_me()



    def update_me(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron()
        if hasattr(self, 'needs_arrow_rebuild') and self.needs_arrow_rebuild:
            self.rebuild_arrows()
            self.needs_arrow_rebuild = False
        #for arrow in self.arrows_forward:
        #    arrow.update_connection()

    def draw_border(self):
        """Draw a rectangle around the perimeter of the display model."""
        pygame.draw.rect(
            self.surface, Const.COLOR_FOR_NEURON_BODY,
            (0, 0, self.width, self.height), 3
        )

    def get_max_blame_for_model(self, run_id: str):
        """
        Retrieves the highest absolute blame value across all epochs and iterations for the given model.

        :param run_id: The model identifier
        :return: The maximum absolute activation value in the run
        """

        SQL_MAX_blame = """
                             SELECT MAX(abs_blame) AS max_abs_blame
                             FROM (SELECT ABS(error_signal) AS abs_blame \
                                   FROM Neuron \
                                   WHERE run_id = ? \
                                   ORDER BY abs_blame ASC LIMIT (SELECT CAST(COUNT(*) * 0.95 AS INT) 
                        FROM Neuron WHERE run_id = ?)) AS FilteredActivations; \
                             """

        result = Const.TRIs[0].db.query(SQL_MAX_blame, (run_id, run_id))
        # print(f"Max activation for run {result}")
        # Return the max activation or a default value to prevent division by zero
        return result[0]['max_abs_blame'] if result and result[0]['max_abs_blame'] is not None else 1.0

    def get_max_activation_for_model(self,  run_id: str):
        """
        Retrieves the highest absolute activation value across all epochs and iterations for the given model.

        :param run_id: The model identifier
        :return: The maximum absolute activation value in the run
        """

        SQL_MAX_ACTIVATION = """
            SELECT MAX(abs_activation) AS max_activation
            FROM (
                SELECT ABS(activation_value) AS abs_activation
                FROM Neuron
                WHERE run_id = ?
                ORDER BY abs_activation ASC
                LIMIT (SELECT CAST(COUNT(*) * 0.95 AS INT) 
                       FROM Neuron WHERE run_id = ?)
            ) AS FilteredActivations;
        """

        result = Const.TRIs[0].db.query(SQL_MAX_ACTIVATION, (run_id,run_id))
        #print(f"Max activation for run {result}")
        # Return the max activation or a default value to prevent division by zero
        return result[0]['max_activation'] if result and result[0]['max_activation'] is not None else 1.0

    def create_output_to_thresholder(self):
        x_start_offset= -3.69
        y_start_offset= 3.69
        x_end_offset= -16.969
        y_end_offset= -3.69
        to_neuron = self.neurons[-1][0] #output neuron
        #to_neuron2 = self.neurons
        #ez_debug(to_neuron=to_neuron)
        #ez_debug(to_neuron2=to_neuron2)
        start_x = to_neuron.location_right_side + x_start_offset
        start_y = to_neuron.location_top+ to_neuron.location_height+ y_start_offset

        if self.prediction_scaler_neuron:
            end_x = self.prediction_scaler_neuron.location_left + x_end_offset+9.6
            end_y = self.prediction_scaler_neuron.neuron_visualizer.my_fcking_labels[0][1]
            self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))


        if self.thresholder :
            # from prediction to threshold
            start = self.prediction_scaler_neuron.neuron_visualizer.label_y_positions[0]
            end = self.thresholder.neuron_visualizer.diamond_left
            self.arrows_forward.append(DisplayArrow(start, end, screen=self.surface,arrow_size=16,thickness=2))

            #from top of diamond to beta class
            start = self.thresholder.neuron_visualizer.diamond_top
            end = self.thresholder.neuron_visualizer.arrow_targets[0]
            self.arrows_forward.append(DisplayArrow(start, end, screen=self.surface,arrow_size=16,thickness=2,label_text="NO"))

            # First leg of alpha class
            start = self.thresholder.neuron_visualizer.diamond_right
            end = self.thresholder.neuron_visualizer.arrow_targets[1]
            self.arrows_forward.append(DisplayArrow(start, end, screen=self.surface,arrow_size=0,thickness=2))

            # Second leg of alpha class
            start = self.thresholder.neuron_visualizer.arrow_targets[1]
            end = self.thresholder.neuron_visualizer.arrow_targets[2]
            self.arrows_forward.append(DisplayArrow(start, end, screen=self.surface, arrow_size=16, thickness=2,label_text="YES"))

    def rebuild_arrows(self):
        """Clear and recreate all arrows based on current neuron positions."""
        # Clear old arrows
        #self.arrows_forward = []

        # Recreate with current positions
        self.create_neuron_to_neuron_arrows(True)

    def create_neuron_to_neuron_arrows(self, forward: bool):
        """Creates neuron-to-neuron arrows using DisplayArrow."""

        self.arrows_forward = []
        y_offset = 10
        #for layer_index in range(1, len(self.config.architecture)):  # Start from the first hidden layer
        for layer_index in range(1, len(self.neurons)):
            current_layer = self.neurons[layer_index - 1]
            next_layer = self.neurons[layer_index]
            for weight_index, from_neuron in enumerate(current_layer):
                for to_neuron in next_layer:
                    if not from_neuron.on_screen or not to_neuron.on_screen: continue # Skip arrows if either neuron is off-screen
                    start_x = from_neuron.location_left + from_neuron.location_width-3.69  # Right edge of from_neuron
                    end_x = to_neuron.location_left  # Left edge of to_neuron

                    # ✅ Ensure we don’t exceed `my_fcking_labels` length
                    max_index_from = min(weight_index + 1, len(from_neuron.neuron_visualizer.my_fcking_labels) - 1)
                    max_index_to = min(weight_index + 1, len(to_neuron.neuron_visualizer.my_fcking_labels) - 1)

                    #start_y = from_neuron.location_top+ from_neuron.location_height//2 + y_offset
                    #start_y = from_neuron.location_top + (from_neuron.location_height * 3) // 4 + y_offset
                    start_y = from_neuron.location_top + from_neuron.location_height  + 3.69
                    end_y = to_neuron.neuron_visualizer.my_fcking_labels[max_index_to][1] + y_offset # Get Y for correct weight

                    self.arrows_forward.append(DisplayArrow(start_x, start_y, end_x, end_y, screen=self.surface))

    #def process_an_event(self, event):
    #    #print(f"my left={self.left} my top = {self.top}")
    #    self.model_button.process_an_event(event)


    def show_info(self):#onclick model button
        print("copy to clipboard")
        """
        import pygame.scrap

        pygame.scrap.init()
        pygame.scrap.set_mode(pygame.SCRAP_CLIPBOARD)
        
        def copy_info_to_clipboard():
            data = your_info_str.encode("utf-8")
            pygame.scrap.put(pygame.SCRAP_TEXT, data)
        """

