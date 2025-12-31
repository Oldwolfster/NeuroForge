# Display_Manager.py

from typing import List
import pygame

from src.NeuroForge import Const
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from src.NNA.engine.RamDB import RamDB
from src.NeuroForge.DisplayBanner import DisplayBanner
from src.NeuroForge.DisplayPanelInput import DisplayPanelInput
from src.NeuroForge.DisplayPanelPrediction import DisplayPanelPrediction
from src.NeuroForge.DisplayPanelTarget import DisplayPanelTarget
from src.NeuroForge.GeneratorModel import ModelGenerator


class Display_Manager:
    """
    Heart of NeuroForge.
    1) Initializes all components: Neurons, UI Panels, Controls
    2) Runs main loops: update, render, event processing
    3) Central location for global data queries
    """

    def __init__(self, TRIs: List[TrainingRunInfo]):
        # Global references
        Const.TRIs = TRIs
        Const.dm = self
        self.db: RamDB = TRIs[0].db

        # Component lists
        self.components = []  # All drawable components
        self.eventors = []  # Components needing event handling
        self.models = []  # Display models
        self.hoverlings = []  # Hover-able objects
        self.the_hovered = None  # Currently hovered object

        # Panel references (set during initialize_components)
        self.input_panel = None
        self.target_panel = None

        # Cached query data
        self.data_sample = None
        self.data_epoch = None
        self.max_blame = 'Unknown'

        # Change detection
        self.last_sample = 0
        self.last_epoch = 0

        # Deferred drawing
        self.deferred_draws: list[tuple[callable, tuple, dict]] = []

        # Initialize
        self.populate_recorded_frames()
        #print(f"recorded_frames count: {len(Const.vcr.recorded_frames)}")
        #print(f"first 5 frames: {Const.vcr.recorded_frames[:5]}")

        self.query_max_values()
        self.query_data_sample()
        self.query_data_epoch()
        self.initialize_components()

    def process_events(self, event): #TODO POPULATE  OR DELETE
        pass

    def render(self):
        """Render all registered components. (Except pop up window"""
        for component in self.components:            #print(f"Rendering: {component.child_name}")  # Print the subclass name
            component.draw_me()

        for fn, args, kwargs in self.deferred_draws:            fn(*args, **kwargs) # run deferred geometry-dependent draws
        self.deferred_draws.clear()

        self.render_popup()

    def render_popup(self):
        """Render tooltips last so they appear above everything."""
        self.update_hover_state()

        if self.the_hovered is not None:
            self.the_hovered.render_tooltip()

    def populate_recorded_frames(self):
        """Load list of all recorded epoch/sample combinations for VCR playback."""
        sql = """
            SELECT epoch, sample 
            FROM RecordSample 
            GROUP BY epoch, sample 
            ORDER BY epoch, sample
        """
        Const.vcr.recorded_frames = self.db.query(sql, as_dict=False)

    def get_max_error_signal_for_iteration(self):
        """
        Get max absolute error_signal for CURRENT sample (sample mode)
        or max of epoch-averaged blames (epoch mode).
        """
        run_id = Const.TRIs[0].run_id

        if Const.vcr.blame_mode == "epoch":
            SQL = """
                SELECT MAX(avg_abs_blame) AS max_error_signal
                FROM (
                    SELECT AVG(ABS(error_signal)) AS avg_abs_blame
                    FROM Neuron 
                    WHERE run_id = ? AND epoch = ?
                    GROUP BY nid
                )
            """
            result = self.db.query(SQL, (run_id, Const.vcr.CUR_EPOCH))
        else:
            SQL = """
                SELECT MAX(ABS(error_signal)) AS max_error_signal
                FROM Neuron WHERE run_id = ? AND epoch = ? AND sample = ? 
            """
            result = self.db.query(SQL, (run_id, Const.vcr.CUR_EPOCH, Const.vcr.CUR_SAMPLE))

        self.max_blame = result[0]['max_error_signal'] if result and result[0]['max_error_signal'] is not None else 1.0
        if not hasattr(self, 'historical_max_blame'):
            self.historical_max_blame = self.max_blame
        else:
            self.historical_max_blame = max(self.historical_max_blame, self.max_blame)

    def query_max_values(self):
        """Query global max values for scaling displays."""
        Const.MAX_EPOCH     = self.db.query_value("SELECT MAX(epoch) FROM RecordSample")
        Const.MAX_SAMPLE    = self.db.query_value("SELECT MAX(sample) FROM RecordSample")
        Const.MAX_WEIGHT    = self.db.query_value("SELECT MAX(ABS(value)) FROM Weight")
        Const.MAX_ERROR     = self.db.query_value("SELECT MAX(ABS(error_signal)) FROM Neuron")

    def query_data_sample(self):
        """Retrieve sample data for each model from the latest valid epoch."""
        sql = """
            SELECT s.*
            FROM RecordSample s
            JOIN (
                SELECT run_id, MAX(epoch) AS latest_epoch
                FROM RecordSample
                WHERE epoch <= ?
                GROUP BY run_id
            ) latest ON s.run_id = latest.run_id AND s.epoch = latest.latest_epoch
            WHERE s.sample = ?
        """
        params = (Const.vcr.CUR_EPOCH, Const.vcr.CUR_SAMPLE)

        rs = self.db.query(sql, params)

        self.data_sample = {row["run_id"]: row for row in rs}


    def query_data_epoch(self):
        """Retrieve epoch summary for each model from the latest valid epoch."""
        sql = """
            SELECT e.*
            FROM RecordEpoch e
            JOIN (
                SELECT run_id, MAX(epoch) AS latest_epoch
                FROM RecordEpoch
                WHERE epoch <= ?
                GROUP BY run_id
            ) latest ON e.run_id = latest.run_id AND e.epoch = latest.latest_epoch
        """
        rs = self.db.query(sql, (Const.vcr.CUR_EPOCH,))
        self.data_epoch = {row["run_id"]: row for row in rs}

        # Display_Manager.py

    def query_data_sample(self):
        """Retrieve sample data for each model from the latest valid epoch."""
        sql = """
            SELECT s.*
            FROM RecordSample s
            JOIN (
                SELECT run_id, MAX(epoch) AS latest_epoch
                FROM RecordSample
                WHERE epoch <= ?
                GROUP BY run_id
            ) latest ON s.run_id = latest.run_id AND s.epoch = latest.latest_epoch
            WHERE s.sample = ?
        """
        params = (Const.vcr.CUR_EPOCH, Const.vcr.CUR_SAMPLE)
        rs = self.db.query(sql, params)
        ###print(f"Const.vcr.CUR_SAMPLE{Const.vcr.CUR_SAMPLE}")
        self.data_sample = {row["run_id"]: row for row in rs}


    def update(self):
        """Main update loop - called each frame."""
        Const.vcr.play_the_tape()

        # Skip if nothing changed
        if self.last_sample == Const.vcr.CUR_SAMPLE and self.last_epoch == Const.vcr.CUR_EPOCH:
            return

        # Something changed - refresh data
        self.last_sample = Const.vcr.CUR_SAMPLE
        self.last_epoch = Const.vcr.CUR_EPOCH

        self.query_data_sample()
        self.query_data_epoch()

        for component in self.components:
            component.update_me()

    def update_hover_state(self):
        """Determine which object is being hovered, prioritizing topmost."""
        self.the_hovered = None
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Global hoverables (buttons, panels)
        for obj in self.hoverlings:
            if obj.is_hovered(0, 0, mouse_x, mouse_y):
                self.the_hovered = obj
                return

        # Model-specific hoverables (neurons, etc.)
        for model in reversed(self.models):  # topmost model first
            for obj in model.hoverlings:
                if obj.is_hovered(model.left, model.top, mouse_x, mouse_y):
                    self.the_hovered = obj
                    return



    def get_sample_data(self, run_id: int) -> dict:
        """Get cached sample data for a specific model."""
        return self.data_sample.get(run_id, {})

    def get_epoch_data(self, run_id: int) -> dict:
        """Get cached epoch data for a specific model."""
        return self.data_epoch.get(run_id, {})

    def initialize_components(self):
        """Initialize all UI components."""
        self.create_banner()
        self.create_left_panels()
        self.create_prediction_panels()
        self.create_models()
        self.register_layer_buttons()
        self.create_outside_arrows()

    def register_layer_buttons(self):
        """Register layer control buttons AFTER models so they render on top."""
        for model in self.models:
            for layer in model.layers:
                for button in layer.control_buttons:
                    if button not in self.components:
                        self.components.append(button)
                        self.eventors.append(button)

    def create_banner(self):
        """Create the top banner showing dataset name, epoch/sample counts."""
        training_data = Const.TRIs[0].training_data
        banner = DisplayBanner(training_data, Const.MAX_EPOCH, Const.MAX_SAMPLE)
        self.components.append(banner)

    def create_left_panels(self):
        """Create the left-side panels: target, inputs, and control."""
        panel_width = 8

        # Target panel
        self.target_panel = DisplayPanelTarget(
            width_pct=panel_width, height_pct=9, left_pct=1, top_pct=10
        )
        self.components.append(self.target_panel)

        # Input panel
        self.input_panel = DisplayPanelInput(
            width_pct=panel_width, height_pct=39, left_pct=1, top_pct=20
        )
        self.components.append(self.input_panel)

    def create_prediction_panels(self):
        panel_width = 8

        for idx, TRI in enumerate(Const.TRIs):
            if idx >= 2:  # Hard limit: 2 models max
                break

            top = 10 if idx == 0 else 52

            panel = DisplayPanelPrediction(
                run_id=TRI.run_id,
                problem_type=TRI.training_data.problem_type,
                TRI=TRI,
                width_pct=panel_width,
                height_pct=39,
                left_pct=99 - panel_width,
                top_pct=top
            )

            self.components.append(panel)

    def create_models(self):
        """Create and register the neural network display models."""
        self.models = ModelGenerator.create_models()
        self.components.extend(self.models)

    def create_outside_arrows(self):
        """Create and register the outside arrows."""

    def schedule_draw(self, fn: callable, *args, **kwargs):
        """Enqueue a draw-call to run after all the regular renders."""
        self.deferred_draws.append((fn, args, kwargs))

