from typing import List
import pygame

from typing import List
import pygame
from itertools import chain
from src.NeuroForge import Const
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from src.NNA.engine.RamDB import RamDB
from src.NeuroForge.ButtonBase import Button_Base
from src.NeuroForge.DisplayArrowsOutsideModel import DisplayArrowsOutsideNeuron
from src.NeuroForge.DisplayBanner import DisplayBanner
from src.NeuroForge.DisplayPanelCtrl import DisplayPanelCtrl
from src.NeuroForge.DisplayPanelInput import DisplayPanelInput
from src.NeuroForge.DisplayPanelPrediction import DisplayPanelPrediction
from src.NeuroForge.DisplayPanelTarget import DisplayPanelTarget
from src.NeuroForge.GeneratorModel import ModelGenerator
from src.NeuroForge.PopupErrorAnalysis import PopupErrorAnalysis
from src.NeuroForge.PopupInfoButton import PopupInfoButton
from src.NeuroForge.PopupTrainingData import PopupTrainingData
from src_revert.NNA.utils.general_text import beautify_text


class Display_Manager:
    """
    This class is the heart of NeuroForge.  It does the following.
    1) Initializes all components including Neurons, UI Panels and Controls, Activations(Outputs)
    2) Runs the main "Loops" such as update, render, and event processing
    3) Is a central location for retrieving and making available global data such as iteration and epoch queries
    It receives all initial needed information in the constructors parameter List[Configs].
    Changing information is retrieved from the in-ram SQL Lite DB that stores the Models states (think VCR Tape of the models training)
    NOTE: The underscore in the class name is deliberate to separate it from the classes it manages (which all are prefixed with 'Display'
    """
    def __init__(self, TRIs: List[TrainingRunInfo]):
        Const.TRIs       = TRIs  # Store all model configs

        self.the_hovered    = None # if an object is being hovered.
        self.hoverlings     = []
        self.t_data_popup   = PopupTrainingData()
        self.info_popup     = PopupInfoButton()
        self.error_analysis_popup = PopupErrorAnalysis()
        self.components     = []  # List for EZSurface-based components
        self.eventors       = []  # Components that need event handling
        self.models         = []  # List for display models
        self.db             = TRIs[0].db  # Temporary shortcut
        #self.bd_label_alpha = TRIs[0].training_data.target_labels[1]
        #self.bd_label_beta  = TRIs[0].training_data.target_labels[0]
        self.data_iteration = None
        self.data_epoch     = None
        self.last_iteration = 0
        self.last_epoch     = 0
        self.input_panel    = None
        self.target_panel   = None
        self.base_window    = None
        Const.dm            = self
        self.max_blame      = 'Unknown'
        self._deferred_draws: list[tuple[callable, tuple, dict]] = [] #collect deferred calls here


        # Compute global max values across all models using Metrics module
        # DELETE ME self.get_max_epoch_per_model(self.db)
        self.populate_list_of_available_frames()
        Const.MAX_EPOCH     = self.get_max_epoch(self.db)
        Const.MAX_SAMPLE = self.get_max_iteration(self.db)
        Const.MAX_WEIGHT    = self.get_max_weight(self.db)
        Const.MAX_ERROR     = self.get_max_error(self.db)

        # Initialize UI Components
        self.query_dict_iteration()
        self.query_dict_epoch()
        self.initialize_components()
        self.get_max_error_signal_for_iteration()

        
        # Add Neurons to each models "hoverlings"
        for model in reversed(self.models):  # ✅ Start with the topmost model
            for layer in model.neurons:
                for neuron in layer:
                    model.hoverlings.append(neuron)
            # Add input scaler neuron if it exists
            if model.input_scaler_neuron: model.hoverlings.append(model.input_scaler_neuron)
            # Add prediction scaler neuron if it exists
            if model.prediction_scaler_neuron: model.hoverlings.append(model.prediction_scaler_neuron)
        self._register_layer_buttons() # After models are created, register all layer buttons

    def _register_layer_buttons(self):
        """Register all layer control buttons with the component system"""
        for model in self.models:
            if hasattr(model, 'layers'):
                layer_buttons = model.get_all_layer_buttons()
                for button in layer_buttons:
                    self.components.append(button)
                    self.hoverlings.append(button)

    def rebuild_model(self):
        """If you rebuild models, re-register buttons"""
        # ... existing model rebuild code ...

        # Clear old buttons from components
        # (You might need to track which are layer buttons to remove only those)

        # Re-register new buttons
        self._register_layer_buttons()


    def populate_list_of_available_frames(self):
        Const.vcr. recorded_frames = self.db.query("SELECT epoch, iteration from Weight group by epoch, iteration order by epoch, iteration",as_dict=False)


    def update(self):
        Const.vcr.play_the_tape()
        if self.last_iteration == Const.vcr.CUR_ITERATION and self.last_epoch == Const.vcr.CUR_EPOCH_MASTER:
            return #No change so no need to update
        self.get_max_error_signal_for_iteration()
        self.last_iteration = Const.vcr.CUR_ITERATION   # Set them to current values
        self.last_epoch     = Const.vcr.CUR_EPOCH_MASTER       # Set them to current values

        for component in self.components:
            component.update_me()


    def get_max_error_signal_for_iteration(self):
        """
        Get max absolute error_signal for CURRENT iteration (sample mode)
        or max of epoch-averaged blames (epoch mode).
        """
        run_id = Const.TRIs[0].run_id  # Get the run_id for this model

        if Const.vcr.blame_mode == "epoch":
            # Get max of epoch-averaged blame values across all neurons
            SQL = """
                SELECT MAX(avg_abs_blame) AS max_error_signal
                FROM (
                    SELECT AVG(ABS(error_signal)) AS avg_abs_blame
                    FROM Neuron 
                    WHERE run_id = ? AND epoch = ?
                    GROUP BY nid
                )
            """
            result = self.db.query(SQL, (run_id, Const.vcr.CUR_EPOCH_MASTER))
        else:
            # Get max for current sample
            SQL = """
                SELECT MAX(ABS(error_signal)) AS max_error_signal
                FROM Neuron WHERE run_id = ? AND epoch = ? AND iteration = ? 
            """
            result = self.db.query(SQL, (run_id, Const.vcr.CUR_EPOCH_MASTER, Const.vcr.CUR_ITERATION))

        self.max_blame = result[0]['max_error_signal'] if result and result[0]['max_error_signal'] is not None else 1.0
        if not hasattr(self, 'historical_max_blame'):
            self.historical_max_blame = self.max_blame
        else:
            self.historical_max_blame = max(self.historical_max_blame, self.max_blame)

    def schedule_draw(self, fn: callable, *args, **kwargs):
        """Enqueue a draw-call to run after all the regular renders."""
        self._deferred_draws.append((fn, args, kwargs))

    def process_events(self, event):
        for component in self.eventors:            #print(f"Display Manager: event={event} ")
            component.process_an_event(event)

    def render(self):
        """Render all registered components. (Except pop up window"""
        for component in self.components:            #print(f"Rendering: {component.child_name}")  # Print the subclass name
            component.draw_me()

        # 2) now execute all the deferred draw calls
        for fn, args, kwargs in self._deferred_draws:
            #print(f"running delayed function{fn}")
            fn(*args, **kwargs)

        # 3) clear the queue for the next frame
        self._deferred_draws.clear()

        # Check if outside arrows need rebuilding (AFTER neurons have rendered and labels repopulated)
        if hasattr(self, 'needs_outside_arrow_rebuild') and self.needs_outside_arrow_rebuild:
            print("🔄 REBUILDING OUTSIDE ARROWS (after render)")
            for component in self.components:
                if hasattr(component, 'rebuild_arrows') and component not in self.models:
                    print(f"  Calling rebuild_arrows on {type(component).__name__}")
                    component.rebuild_arrows()
            self.needs_outside_arrow_rebuild = False
            print("✅ Outside arrows rebuilt")

    def render_pop_up_window(self):
        """
        This is rendered separately to ensure it is last and is not overwritten by anything
        such as UI Controls.
        """

        #if Const.tool_tip_to_show is not None:
        #    print(f"Showing popup - Const.tool_tip_to_show = {Const.tool_tip_to_show }")
        #    Const.tool_tip_to_show()
        #    #Const.tool_tip_to_show = None


        #print(f"Const.tool_tip_to_show = {Const.tool_tip_to_show}")
        #if Const.tool_tip_to_show is not None:
        #    Const.tool_tip_to_show()   # now calls render()
        #    Const.tool_tip_to_show = None

        self.update_hover_state()
        if self.the_hovered is not None:
            self.the_hovered.render_tooltip()


    def initialize_components(self):
        """Initialize UI components like EZForm-based input panels and model displays."""
        display_banner = DisplayBanner(Const.TRIs[0].training_data, Const.MAX_EPOCH, Const.MAX_SAMPLE)
        self.components.append(display_banner)
        panel_width = 8

        # Render behind the target button
        button_td=Button_Base(text=beautify_text("Training Data"),
                      width_pct=panel_width-.5, height_pct=8, left_pct=1.5, top_pct=11, on_click=self.show_info,
                      on_hover=lambda: self.t_data_popup.show_me(),
                      shadow_offset=-5, auto_size=False, my_surface=Const.SCREEN,
                      text_line2=f"(click for details)", surface_offset=(0, 0))
        self.components.append(button_td)
        self.hoverlings.append(button_td)

        #Second button behind inputs
        button_td=Button_Base(text=beautify_text("Training Data"),
                      width_pct=panel_width-.5, height_pct=39, left_pct=1.5, top_pct=20, on_click=self.show_info,
                      on_hover=lambda: self.t_data_popup.show_me(),
                      shadow_offset=-5, auto_size=False, my_surface=Const.SCREEN,
                      text_line2=f"(click for details)", surface_offset=(0, 0))
        self.components.append(button_td)
        self.hoverlings.append(button_td)



        button_info = Button_Base(my_surface= Const.SCREEN, text="Sample",
                                  width_pct=panel_width, height_pct=4, left_pct=1, top_pct=5,
                                  on_click=self.show_info,
                                  on_hover=lambda: self.info_popup.show_me(),
                                  shadow_offset=-5)
        self.components.append(button_info)
        self.hoverlings.append(button_info)


        # Add Target Panel  # Storing reference for arrows from input to first layer of neurons
        self.target_panel = DisplayPanelTarget(width_pct=panel_width, height_pct=9, left_pct=1, top_pct=10)
        self.components.append(self.target_panel)


        # Add Input Panel  # Storing reference for arrows from input to first layer of neurons
        self.input_panel = DisplayPanelInput(width_pct=panel_width, height_pct=39, left_pct=1, top_pct=20)
        self.components.append(self.input_panel)

        # Add Control Panel
        panel = DisplayPanelCtrl( width_pct=panel_width, height_pct=34, left_pct=1, top_pct=60)
        self.components.append(panel)
        self.eventors.append(panel)

        # Add Prediction Panels for each model
        self.create_prediction_panels(panel_width)

        # Create Models
        self.models = ModelGenerator.create_models()
        self.components.extend(self.models) #add models to component list

        # Add Input and output Arrows (Spans multiple surfaces) - will be full area and not clear)
        arrows = DisplayArrowsOutsideNeuron(self.models[0], True)
        self.components.append(arrows)
        if len(self.models) > 1:

            arrows = DisplayArrowsOutsideNeuron(self.models[1], False)
            self.components.append(arrows)

        # Add window Match
        #self.base_window = BaseWindow(width_pct=60, height_pct=60, left_pct=20, top_pct=15, banner_text="Configure Match",
        #                              background_image_path="assets/form_backgrounds/coliseum_glow.png")
        #win_matches = WindowMatches()
        #self.components.append(win_matches)
        #self.eventors.append(win_matches)

    def show_info(self):
        print("Show info")

    def create_prediction_panelsOrig(self, panel_width): #one needed per model
        for idx, TRI in enumerate(Const.TRIs):
            run_id = TRI.run_id  # Assuming Config has a `run_id` attribute
            problem_type = TRI.training_data.problem_type
            #For now, this will show 2 and write the rest over the top of each other.
            top = 10 #Assume 1 model
            if idx == 1:    #move 2nd box down (0 based)
                top = 52
            if idx <2:      #Only show two prediction panels
                panel = DisplayPanelPrediction(run_id, problem_type, TRI, width_pct=panel_width, height_pct=39, left_pct=99-panel_width, top_pct=top)
                self.add_error_analysis_button(width_pct=panel_width, height_pct=39, left_pct=99 - panel_width,                                               top_pct=top)
                self.components.append(panel)


    def add_error_analysis_buttonOrig(self, width_pct, height_pct, left_pct, top_pct):
        """
        Creates an invisible button over the prediction panel to trigger Error Analysis popup.
        Coordinates match the prediction panel exactly so hover works intuitively.
        """
        button = Button_Base(
            text="",  # Invisible
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            on_click=None,
            on_hover=lambda: self.error_analysis_popup.show_me(),
            shadow_offset=0,
            auto_size=False,
            my_surface=Const.SCREEN,
            surface_offset=(0, 0)
        )
        self.hoverlings.append(button)
        self.components.append(button)

    # Display_Manager.py

    def create_prediction_panels(self, panel_width):
        for idx, TRI in enumerate(Const.TRIs):
            run_id = TRI.run_id
            problem_type = TRI.training_data.problem_type
            top = 10
            if idx == 1:
                top = 52
            if idx < 2:
                panel = DisplayPanelPrediction(run_id, problem_type, TRI, width_pct=panel_width, height_pct=39,
                                               left_pct=99 - panel_width, top_pct=top)
                self.add_error_analysis_button(width_pct=panel_width, height_pct=39, left_pct=99 - panel_width,
                                               top_pct=top, TRI=TRI)  # Pass TRI!
                self.components.append(panel)

    def add_error_analysis_button(self, width_pct, height_pct, left_pct, top_pct, TRI):
        """
        Creates an invisible button over the prediction panel to trigger Error Analysis popup.
        Coordinates match the prediction panel exactly so hover works intuitively.
        """
        # Find the index of this TRI in Const.TRIs
        model_idx = Const.TRIs.index(TRI)

        button = Button_Base(
            text="",  # Invisible
            width_pct=width_pct,
            height_pct=height_pct,
            left_pct=left_pct,
            top_pct=top_pct,
            on_click=None,
            on_hover=lambda: self.show_error_analysis_for_model(model_idx),  # Closure captures model_idx!
            shadow_offset=0,
            auto_size=False,
            my_surface=Const.SCREEN,
            surface_offset=(0, 0)
        )
        self.hoverlings.append(button)
        self.components.append(button)

    def show_error_analysis_for_model(self, model_idx: int):
        """Show error analysis popup for specific model"""
        self.error_analysis_popup.set_model_index(model_idx)
        self.error_analysis_popup.show_me()


    def query_dict_iteration(self):
        """Retrieve iteration data for each model from the latest valid epoch."""
        sql = """
            SELECT i.*
            FROM Iteration i
            JOIN (
                SELECT run_id, MAX(epoch) AS latest_epoch
                FROM Iteration
                WHERE epoch <= ?
                GROUP BY run_id
            ) latest ON i.run_id = latest.run_id AND i.epoch = latest.latest_epoch
            WHERE i.iteration = ?
        """
        params = (Const.vcr.CUR_EPOCH_MASTER, Const.vcr.CUR_ITERATION)
        rs = self.db.query(sql, params)

        self.data_iteration = {row["run_id"]: row for row in rs}

    def query_dict_epoch(self):
        sql = """
            SELECT e.*
            FROM EpochSummary e
            JOIN (
                SELECT run_id, MAX(epoch) AS latest_epoch
                FROM EpochSummary
                WHERE epoch <= ?
                GROUP BY run_id
            ) latest ON e.run_id = latest.run_id AND e.epoch = latest.latest_epoch
        """
        rs = self.db.query(sql, (Const.vcr.CUR_EPOCH_MASTER,))
        self.data_epoch = {row["run_id"]: row for row in rs}

    def get_model_iteration_data(self, run_id: str ) -> dict:
        """Retrieve iteration data for a specific model from the cached dictionary."""
        # REMOVED DEFAULT BECAUSE MAKES IT EASY TO CREATE SILENT HARD TO FIND BUGS
        #ez_debug(run_id_from_dm=run_id)
        if run_id!="any":
            return self.data_iteration.get(run_id, {})

        # If no run_id is provided, return the first available model's data
        for model in self.data_iteration.values():
            return model  # Return the first entry found
        return {}

    def get_model_epoch_data(self, run_id: str = None) -> dict:
        """Retrieve iteration data for a specific model from the cached dictionary."""
        if run_id:
            return self.data_epoch.get(run_id, {})

        # If no run_id is provided, return the first available model's data
        for model in self.data_epoch.values():
            return model  # Return the first entry found
        return {}

    def get_max_error(self, db: RamDB) -> int:
        """Retrieve highest abs(error)"""
        sql = "SELECT MAX(abs(error_signal)) as error_signal FROM Neuron"
        rs = db.query(sql)
        return rs[0].get("error_signal")

    def get_max_epoch(self, db: RamDB) -> int:
        """Retrieve highest epoch for all models."""

        sql = "SELECT MAX(epoch) as max_epoch FROM Iteration"
        rs = db.query(sql)
        return rs[0].get("max_epoch")

    def get_max_weight(self, db: RamDB) -> float:
        """Retrieve highest weight magnitude."""
        sql = "SELECT MAX(ABS(value)) AS max_weight FROM Weight"
        rs = db.query(sql)
        return rs[0].get("max_weight")

    def get_max_iteration(self, db: RamDB) -> int:
        """Retrieve highest iteration"""
        sql = "SELECT MAX(iteration) as max_iteration FROM Iteration"
        rs = db.query(sql)
        return rs[0].get("max_iteration")

    def update_hover_state(self):
        """
        Check which neuron is being hovered over, prioritizing the topmost model.
        """
        self.the_hovered = None  # Reset each frame
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for obj in self.hoverlings:
            if obj.is_hovered(0,0,mouse_x,mouse_y):
                self.the_hovered = obj
                return

        for model in reversed(self.models):  # ✅ Start with the topmost model
            for potential in model.hoverlings:
                if potential.is_hovered(model.left, model.top, mouse_x, mouse_y):
                    self.the_hovered = potential    # We found one, store it
                    return                                     # ✅ Stop checking once we find one (avoids conflicts)
