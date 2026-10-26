import time
import pygame
from src.NeuroForge import Const
from src.NeuroForge.RemoteNeuronReports import RemoteNeuronReports


class HoldTimerManager:
    def __init__(self):
        self.hold_start_time = None
        self.hold_neuron_context = None
        self.neuron_report_remote = None


    def update(self, the_hovered):

        mouse_buttons = pygame.mouse.get_pressed()
        mouse_down = mouse_buttons[0]
        #print(f"DEBUG: the_hovered={the_hovered}, mouse_down={pygame.mouse.get_pressed()[0]}")  # ← ADD THIS
        #return #TODO Get this working - breaks when you hit a nav buttton
        if the_hovered is not None and mouse_down  and hasattr(the_hovered, 'nid'):
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
                self.hold_neuron_context = {'nid': the_hovered.nid}
            else:
                elapsed = time.time() - self.hold_start_time
                if elapsed >= 1.69:
                    self.trigger_report_remote()
        else:
            self.hold_start_time = None
            self.hold_neuron_context = None

    def trigger_report_remote(self):
        if self.hold_neuron_context is not None:
            ctx = self.hold_neuron_context
            self.neuron_report_remote = RemoteNeuronReports(nid=ctx['nid'])
        self.hold_start_time = None
        self.hold_neuron_context = None

    def has_active_remote(self):
        return self.neuron_report_remote is not None

    def close_remote(self):
        self.neuron_report_remote = None

    def process_remote_event(self, event):
        if self.neuron_report_remote is not None:
            self.neuron_report_remote.handle_event(event)

    def render_remote(self):
        if self.neuron_report_remote is not None:
            self.neuron_report_remote.render()

    def render_hold_progress(self):
        if self.hold_start_time is None:
            return

        elapsed = time.time() - self.hold_start_time
        progress = min(elapsed / 1.69, 1.0)

        mouse_x, mouse_y = pygame.mouse.get_pos()
        center_x = mouse_x + 40
        center_y = mouse_y - 40
        radius = 30

        circle_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        circle_center = (radius + 2, radius + 2)

        pygame.draw.circle(circle_surf, (60, 60, 60, 200), circle_center, radius)

        if progress > 0:
            end_angle = -90 + (360 * progress)
            points = [circle_center]
            for angle in range(-90, int(end_angle) + 1, 5):
                rad = angle * 3.14159 / 180
                px = circle_center[0] + int(radius * 0.8 * pygame.math.Vector2(1, 0).rotate(angle + 90).x)
                py = circle_center[1] + int(radius * 0.8 * pygame.math.Vector2(1, 0).rotate(angle + 90).y)
                points.append((px, py))

            if len(points) > 2:
                pygame.draw.polygon(circle_surf, (100, 150, 255, 220), points)

        pygame.draw.circle(circle_surf, (200, 200, 200), circle_center, radius, 2)

        Const.SCREEN.blit(circle_surf, (center_x - radius - 2, center_y - radius - 2))