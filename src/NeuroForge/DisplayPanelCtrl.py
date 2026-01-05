from enum import Enum, auto
import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZFormLEFT import EZForm
from src.NeuroForge.TextBox import TextBox
from src.NeuroForge.ButtonBase import Button_Base

class Action(Enum):
    def __new__(cls, desc: str, key: str = ""):
        obj         = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1
        obj.desc    = desc
        obj.key     = key
        return obj


    STEP_BACK_SAMPLE        = ("Step back one sample"       , "Q")
    STEP_FORWARD_SAMPLE     = ("Step forward one sample"    , "W")
    STEP_BACK_EPOCH         = ("Step back one epoch"        , "A")
    STEP_FORWARD_EPOCH      = ("Step forward one epoch"     , "S")
    STEP_BACK_BIG           = ("Step back 100 epochs"       , "Z")
    STEP_FORWARD_BIG        = ("Step forward 100 epochs"    , "X")
    TOGGLE_PLAY             = ("Play / Pause"               , "Tab")
    TOGGLE_REVERSE          = ("Toggle direction"           , "E")
    TOGGLE_BLAME_MODE       = ("Toggle blame mode"          , "B")
    CYCLE_SPEED             = ("Cycle playback speed"       , "")
    JUMP_TO_EPOCH           = ("Jump to specific epoch"     , "")
    JUMP_TO_SAMPLE          = ("Jump to specific sample"    , "")


class DisplayPanelCtrl(EZForm):
    SPEED_OPTIONS = ["1x", "2x", "4x", "10x", "25x", "50x", "Iter"]

    def __init__(self, width_pct: float, height_pct: float, left_pct: float, top_pct: float):
        super().__init__({}, width_pct, height_pct, left_pct, top_pct, banner_text="Controls")

        self.buttons = {}
        self.speed_index = 0
        self.create_buttons()



    def create_buttons(self):
        """Create all control buttons using percentage-based positioning."""
        # Layout constants (percentages of PANEL size)
        left_offset = 6  # Extra space on left for shadow
        side_margin = 4  # Left and right padding
        top_margin = 12  # Top padding (below banner)
        row_gap = 2  # Space between rows
        btn_gap = 4  # Space between two buttons on same row
        num_rows = 8  # Total button rows

        # Calculate button dimensions
        left_edge = left_offset + side_margin
        right_edge = 100 - side_margin
        available_width = right_edge - left_edge
        full_btn_width = available_width
        half_btn_width = (available_width - btn_gap) / 2

        available_height = 100 - top_margin - side_margin
        total_gaps = (num_rows - 1) * row_gap
        btn_height = (available_height - total_gaps) / num_rows

        # Column positions - calculate col2 from right edge to avoid jagged right side
        col1_left = left_edge
        col2_left = right_edge - half_btn_width

        def row_top(row_num):
            return top_margin + row_num * (btn_height + row_gap)

        def add_button(action, text, col, row, full_width=False, text_line2=None):
            tooltip = action.desc
            if action.key:
                tooltip += f"  [{action.key}]"

            if full_width:
                left = col1_left
                width = full_btn_width
            else:
                left = col1_left if col == 0 else col2_left
                width = half_btn_width

            btn = Button_Base(
                my_surface=self.surface,
                text=text,
                left_pct=left,
                top_pct=row_top(row),
                width_pct=width,
                height_pct=btn_height,
                on_click=lambda a=action: self.perform_action(a),
                font_size=20,
                shadow_offset=-3,
                text_line2=text_line2,
                text_line2_color=Const.COLOR_GRAY if text_line2 else None,
                surface_offset=(self.left, self.top),
                tooltip_text=tooltip,
            )
            self.buttons[action] = btn
            return btn

        # Row 0: Speed (full width)
        add_button(Action.CYCLE_SPEED, f"Speed: {self.SPEED_OPTIONS[self.speed_index]}", 0, 0, full_width=True)

        # Row 1: Pause | Reverse
        add_button(Action.TOGGLE_PLAY, "Pause", 0, 1)
        add_button(Action.TOGGLE_REVERSE, "Reverse", 1, 1)

        # Row 2: < | >
        add_button(Action.STEP_BACK_SAMPLE, "<", 0, 2)
        add_button(Action.STEP_FORWARD_SAMPLE, ">", 1, 2)

        # Row 3: << | >>
        add_button(Action.STEP_BACK_EPOCH, "<<", 0, 3)
        add_button(Action.STEP_FORWARD_EPOCH, ">>", 1, 3)

        # Row 4: <<<< | >>>>
        add_button(Action.STEP_BACK_BIG, "<<<<", 0, 4)
        add_button(Action.STEP_FORWARD_BIG, ">>>>", 1, 4)


        # Row 5: Jump to Epoch (full width)
        self.txt_sample = TextBox(
            my_surface=self.surface,
            left_pct=col1_left,
            top_pct=row_top(6),
            width_pct=full_btn_width,
            height_pct=btn_height,
            placeholder="Jump to Sample",
            prompt="Enter Sample #",
            on_submit=lambda val: Const.vcr.jump_to_sample(int(val)),
            numeric_only=True,
            surface_offset=(self.left, self.top),
        )

        self.txt_epoch = TextBox(
            my_surface=self.surface,
            left_pct=col1_left,
            top_pct=row_top(7),
            width_pct=full_btn_width,
            height_pct=btn_height,
            placeholder="Jump to Epoch",
            prompt="Enter Epoch #",
            on_submit=lambda val: Const.vcr.jump_to_epoch(int(val)),
            numeric_only=True,
            surface_offset=(self.left, self.top),
        )
        # Row 7: Blame mode (full width)
        add_button(Action.TOGGLE_BLAME_MODE, f"Blame: {Const.vcr.blame_mode.title()}", 0, 5, full_width=True)

        for btn in self.buttons.values():
            Const.dm.hoverlings.append(btn)

    def perform_action(self, action: Action):
        if action == Action.TOGGLE_PLAY:
            self.toggle_playback()
        elif action == Action.TOGGLE_REVERSE:
            self.toggle_reverse()
        elif action == Action.TOGGLE_BLAME_MODE:
            self.toggle_blame_mode()
        elif action == Action.CYCLE_SPEED:
            self.cycle_speed()
        elif action == Action.STEP_BACK_SAMPLE:
            Const.vcr.step_x_samples(-1, True)
        elif action == Action.STEP_FORWARD_SAMPLE:
            Const.vcr.step_x_samples(1, True)
        elif action == Action.STEP_BACK_EPOCH:
            Const.vcr.step_x_epochs(-1, True)
        elif action == Action.STEP_FORWARD_EPOCH:
            Const.vcr.step_x_epochs(1, True)
        elif action == Action.STEP_BACK_BIG:
            Const.vcr.step_x_epochs(-100, True)
        elif action == Action.STEP_FORWARD_BIG:
            Const.vcr.step_x_epochs(100, True)
        elif action == Action.JUMP_TO_EPOCH:
            pass  # TODO: implement popup or input
        elif action == Action.JUMP_TO_SAMPLE:
            pass  # TODO: implement popup or input

    def toggle_playback(self):
        if Const.vcr.status == "Playing":
            Const.vcr.pause()
            self.buttons[Action.TOGGLE_PLAY].text = "Play"
        else:
            Const.vcr.play()
            self.buttons[Action.TOGGLE_PLAY].text = "Pause"

    def toggle_reverse(self):
        Const.vcr.reverse()
        is_reversed = Const.vcr.direction == -1
        self.buttons[Action.TOGGLE_REVERSE].text = "Forward" if is_reversed else "Reverse"

    def toggle_blame_mode(self):
        Const.vcr.blame_mode = "sample" if Const.vcr.blame_mode == "epoch" else "epoch"
        self.buttons[Action.TOGGLE_BLAME_MODE].text = f"Blame: {Const.vcr.blame_mode.title()}"

    def cycle_speed(self):
        self.speed_index = (self.speed_index + 1) % len(self.SPEED_OPTIONS)
        speed = self.SPEED_OPTIONS[self.speed_index]
        self.buttons[Action.CYCLE_SPEED].text = f"Speed: {speed}"

        if speed == "Iter":
            Const.vcr.advance_by_epoch = 0
            Const.vcr.set_speed(1)
        else:
            Const.vcr.advance_by_epoch = 1
            Const.vcr.set_speed(int(speed.replace("x", "")))

    def process_an_event(self, event):
        # Pass events to all buttons
        for button in self.buttons.values():
            button.process_an_event(event)
        self.txt_epoch .process_an_event(event)
        self.txt_sample.process_an_event(event)

        # Keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            self.process_keyboard_event(event)

    def process_keyboard_event(self, event):
        key_map = {
            pygame.K_q: Action.STEP_BACK_SAMPLE,
            pygame.K_w: Action.STEP_FORWARD_SAMPLE,
            pygame.K_a: Action.STEP_BACK_EPOCH,
            pygame.K_s: Action.STEP_FORWARD_EPOCH,
            pygame.K_z: Action.STEP_BACK_BIG,
            pygame.K_x: Action.STEP_FORWARD_BIG,
            pygame.K_TAB: Action.TOGGLE_PLAY,
            pygame.K_e: Action.TOGGLE_REVERSE,
            pygame.K_b: Action.TOGGLE_BLAME_MODE,
            pygame.K_ESCAPE: None,  # Handled separately
        }

        if event.key == pygame.K_ESCAPE:
            Const.IS_RUNNING = False
            return

        action = key_map.get(event.key)
        if action:
            self.perform_action(action)

    def draw_me(self):
        self.render()  # EZForm draws banner/background to self.surface
        for button in self.buttons.values():
            button.draw_me()  # Buttons draw to self.surface
        self.txt_epoch .draw_me()
        self.txt_sample.draw_me()
        Const.SCREEN.blit(self.surface, (self.left, self.top)) # Blit the surface to screen


    def update_me(self):
        # Sync button text with VCR state
        if Const.vcr.status == "Playing":
            self.buttons[Action.TOGGLE_PLAY].text = "Pause"
        else:
            self.buttons[Action.TOGGLE_PLAY].text = "Play"

        self.buttons[Action.TOGGLE_BLAME_MODE].text = f"Blame: {Const.vcr.blame_mode.title()}"