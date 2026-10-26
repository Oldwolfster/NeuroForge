# VCR.py

import time
from bisect import bisect_left
from src.NeuroForge import Const


class VCR:
    def __init__(self):
        self.vcr_rate = 5  # Frames per second
        self.direction = 1  # 1 = Forward, -1 = Reverse
        self.advance_by_epoch = 1  # 1 = epoch mode, 0 = sample mode
        self.last_update_time = time.monotonic()
        self.status = "Playing"
        self.recorded_frames = []  # Filled from Display_Manager
        self._cur_epoch = 1
        self._CUR_SAMPLE = 1
        self.blame_mode = "epoch"  # "epoch" = averaged, "sample" = per-sample

    def get_nearest_frame(self, requested_epoch, requested_iter):
        """Snap to nearest recorded frame - handles partial records gracefully"""
        requested = (requested_epoch, requested_iter)
        frames = self.recorded_frames

        if not frames:
            return requested
        if requested in frames:
            return requested

        idx = bisect_left(frames, requested)

        if idx < len(frames):
            return frames[idx]
        elif idx > 0:
            return frames[idx - 1]
        else:
            return frames[0]

    @property
    def CUR_EPOCH(self):
        return self._cur_epoch

    @CUR_EPOCH.setter
    def CUR_EPOCH(self, val):
        e, i = self.get_nearest_frame(val, self._CUR_SAMPLE)
        self._cur_epoch = e
        self._CUR_SAMPLE = i

    @property
    def CUR_SAMPLE(self):
        return self._CUR_SAMPLE

    @CUR_SAMPLE.setter
    def CUR_SAMPLE(self, val):
        e, i = self.get_nearest_frame(self._cur_epoch, val)
        self._cur_epoch = e
        self._CUR_SAMPLE = i

    @property
    def now(self):
        return f"Epoch:{self._cur_epoch} Sample:{self._CUR_SAMPLE}"

    def play(self):
        self.status = "Playing"

    def pause(self):
        self.status = "Paused"

    def toggle_play_pause(self):
        if self.status == "Playing":
            self.pause()
        else:
            self.play()

    def reverse(self):
        self.direction *= -1
        self.play()

    def set_speed(self, speed: int):
        self.vcr_rate = abs(speed) * self.direction
        if speed == 0:
            self.pause()
        else:
            self.play()

    def jump_to_epoch(self, epoch_str: str):
        """Jump to specific epoch, snap to nearest recorded frame"""
        try:
            target_epoch = int(epoch_str)
            if 1 <= target_epoch <= Const.MAX_EPOCH:
                self.pause()
                snapped_epoch, snapped_iter = self.get_nearest_frame(target_epoch, self._CUR_SAMPLE)
                self._cur_epoch = snapped_epoch
                self._CUR_SAMPLE = snapped_iter
                self.validate_and_sync()
            else:
                print(f"⚠️ Epoch out of range! Must be between 1 and {Const.MAX_EPOCH}.")
        except ValueError:
            print("⚠️ Invalid input! Please enter a valid epoch number.")

    def jump_to_sample(self, sample_str: str):
        """Jump to specific sample, snap to nearest recorded frame"""
        try:
            target_sample = int(sample_str)
            if 1 <= target_sample <= Const.MAX_SAMPLE:
                self.pause()
                snapped_epoch, snapped_iter = self.get_nearest_frame(self._cur_epoch, target_sample)
                self._cur_epoch = snapped_epoch
                self._CUR_SAMPLE = snapped_iter
                self.validate_and_sync()
            else:
                print(f"⚠️ Sample out of range! Must be between 1 and {Const.MAX_SAMPLE}.")
        except ValueError:
            print("⚠️ Invalid input! Please enter a valid sample number.")

    def step_x_samples(self, step: int, pause_me=False):
        """Move specified iterations forward or backward"""
        if pause_me:
            self.pause()

        if step < 0 and Const.vcr.CUR_SAMPLE == 1:
            if Const.vcr.CUR_EPOCH > 1:
                Const.vcr.CUR_EPOCH -= 1
                Const.vcr.CUR_SAMPLE = Const.MAX_SAMPLE
            return
        elif step > 0 and Const.vcr.CUR_SAMPLE == Const.MAX_SAMPLE:
            if Const.vcr.CUR_EPOCH < Const.MAX_EPOCH:
                Const.vcr.CUR_EPOCH += 1
                Const.vcr.CUR_SAMPLE = 1
            return

        Const.vcr.CUR_SAMPLE += step
        snapped_epoch, snapped_iter = self.get_nearest_frame(Const.vcr.CUR_EPOCH, Const.vcr.CUR_SAMPLE)
        Const.vcr.CUR_EPOCH = snapped_epoch
        Const.vcr.CUR_SAMPLE = snapped_iter
        self.validate_and_sync()

    def step_x_epochs(self, step: int, pause_me=False):
        """Move specified epochs forward or backward"""
        if pause_me:
            self.pause()
        Const.vcr.CUR_EPOCH += step
        self.validate_and_sync()

    def play_the_tape(self):
        """Auto-advance when playing"""
        if self.status == "Playing":
            current_time = time.monotonic()
            seconds_per_frame = 1.0 / abs(self.vcr_rate)
            if current_time - self.last_update_time >= seconds_per_frame:
                self.switch_frame()
                self.last_update_time = current_time

    def switch_frame(self):
        """Advance or reverse based on playback direction"""
        if self.status != "Playing":
            return
        if self.advance_by_epoch == 1:
            self.step_x_epochs(self.direction)
        else:
            self.step_x_samples(self.direction)

    def validate_and_sync(self):
        """Keep values in bounds and refresh display data"""
        if Const.vcr.CUR_EPOCH > Const.MAX_EPOCH:
            Const.vcr.CUR_EPOCH = Const.MAX_EPOCH
            Const.vcr.CUR_SAMPLE = Const.MAX_SAMPLE
            self.pause()

        if Const.vcr.CUR_EPOCH < 1:
            Const.vcr.CUR_EPOCH = 1
            Const.vcr.CUR_SAMPLE = 1
            self.pause()

        if Const.vcr.CUR_SAMPLE > Const.MAX_SAMPLE:
            Const.vcr.CUR_EPOCH = min(Const.vcr.CUR_EPOCH + 1, Const.MAX_EPOCH)
            Const.vcr.CUR_SAMPLE = 1

        if Const.vcr.CUR_SAMPLE < 1:
            Const.vcr.CUR_EPOCH = max(1, Const.vcr.CUR_EPOCH - 1)
            Const.vcr.CUR_SAMPLE = Const.MAX_SAMPLE

        Const.dm.query_data_sample()
        Const.dm.query_data_epoch()