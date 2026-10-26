import pygame

from src.NeuroForge import Const

REPORT_META = {
    1: {"name": "Report 1", "desc": "report1_meta - placeholder"},
    2: {"name": "Report 2", "desc": "report2_meta - placeholder"},
    3: {"name": "Report 3", "desc": "report3_meta - placeholder"},
    4: {"name": "Report 4", "desc": "report4_meta - placeholder"},
    5: {"name": "Report 5", "desc": "report5_meta - placeholder"},
    6: {"name": "Report 6", "desc": "report6_meta - placeholder"},
    7: {"name": "Report 7", "desc": "report7_meta - placeholder"},
    8: {"name": "Report 8", "desc": "report8_meta - placeholder"},
    9: {"name": "Report 9", "desc": "report9_meta - placeholder"},
    10: {"name": "Report 10", "desc": "report10_meta - placeholder"},
}


class RemoteNeuronReports:
    def __init__(self, nid):
        self.nid = nid
        self.selected_id = 1

        screen_w, screen_h = Const.SCREEN.get_size()

        self.modal_w = int(screen_w * 0.70)
        self.modal_h = int(screen_h * 0.80)
        self.modal_x = (screen_w - self.modal_w) // 2
        self.modal_y = (screen_h - self.modal_h) // 2

        self.rect = pygame.Rect(self.modal_x, self.modal_y, self.modal_w, self.modal_h)

        self.left_panel_w = int(self.modal_w * 0.25)
        self.right_panel_w = self.modal_w - self.left_panel_w
        self.button_h = int(self.modal_h * 0.07)

        self.report_button_rects = []
        self.build_report_buttons()
        self.build_action_buttons()

    def build_report_buttons(self):
        start_x = self.modal_x + 10
        start_y = self.modal_y + 50

        for i in range(1, 11):
            btn_rect = pygame.Rect(
                start_x,
                start_y + (i - 1) * (self.button_h + 5),
                self.left_panel_w - 20,
                self.button_h
            )
            self.report_button_rects.append(btn_rect)

    def build_action_buttons(self):
        btn_w = 120
        btn_h = 40
        spacing = 20
        bottom_y = self.modal_y + self.modal_h - btn_h - 20

        cancel_x = self.modal_x + self.modal_w - btn_w - spacing
        run_x = cancel_x - btn_w - spacing

        self.btn_run_rect = pygame.Rect(run_x, bottom_y, btn_w, btn_h)
        self.btn_cancel_rect = pygame.Rect(cancel_x, bottom_y, btn_w, btn_h)

    def select_report(self, report_id):
        self.selected_id = report_id

    def run_report(self):
        print(f"Running Report {self.selected_id} for neuron nid={self.nid}")
        self.close()

    def close(self):
        from Display_Manager import Display_Manager
        Const.dm.close_remote()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, rect in enumerate(self.report_button_rects):
                if rect.collidepoint(event.pos):
                    self.select_report(i + 1)
                    return

            if self.btn_run_rect.collidepoint(event.pos):
                self.run_report()
                return
            if self.btn_cancel_rect.collidepoint(event.pos):
                self.close()
                return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.close()

    def render(self):
        overlay = pygame.Surface(Const.SCREEN.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        Const.SCREEN.blit(overlay, (0, 0))

        pygame.draw.rect(Const.SCREEN, (40, 40, 40), self.rect)
        pygame.draw.rect(Const.SCREEN, (200, 200, 200), self.rect, 2)

        self.render_title()
        self.render_report_buttons()
        self.render_description()
        self.render_action_buttons()

    def render_title(self):
        font = pygame.font.SysFont("Arial", 24, bold=True)
        title_text = f"Reports for Neuron (nid={self.nid})"
        title_surf = font.render(title_text, True, (255, 255, 255))
        title_x = self.modal_x + 20
        title_y = self.modal_y + 15
        Const.SCREEN.blit(title_surf, (title_x, title_y))

    def render_report_buttons(self):
        font = pygame.font.SysFont("Arial", 18)

        for i, rect in enumerate(self.report_button_rects):
            if i + 1 == self.selected_id:
                pygame.draw.rect(Const.SCREEN, (100, 150, 255), rect)
            else:
                pygame.draw.rect(Const.SCREEN, (60, 60, 60), rect)

            pygame.draw.rect(Const.SCREEN, (200, 200, 200), rect, 1)

            text_surf = font.render(f"Report {i + 1}", True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=rect.center)
            Const.SCREEN.blit(text_surf, text_rect)

    def render_description(self):
        desc_x = self.modal_x + self.left_panel_w + 20
        desc_y = self.modal_y + 60

        meta = REPORT_META[self.selected_id]

        font_title = pygame.font.SysFont("Arial", 20, bold=True)
        font_desc = pygame.font.SysFont("Arial", 16)

        title_surf = font_title.render(meta["name"], True, (255, 255, 255))
        desc_surf = font_desc.render(meta["desc"], True, (200, 200, 200))

        Const.SCREEN.blit(title_surf, (desc_x, desc_y))
        Const.SCREEN.blit(desc_surf, (desc_x, desc_y + 35))

    def render_action_buttons(self):
        font = pygame.font.SysFont("Arial", 20)

        pygame.draw.rect(Const.SCREEN, (50, 150, 50), self.btn_run_rect)
        pygame.draw.rect(Const.SCREEN, (200, 200, 200), self.btn_run_rect, 2)
        run_text = font.render("Run Report", True, (255, 255, 255))
        run_rect = run_text.get_rect(center=self.btn_run_rect.center)
        Const.SCREEN.blit(run_text, run_rect)

        pygame.draw.rect(Const.SCREEN, (150, 50, 50), self.btn_cancel_rect)
        pygame.draw.rect(Const.SCREEN, (200, 200, 200), self.btn_cancel_rect, 2)
        cancel_text = font.render("Cancel", True, (255, 255, 255))
        cancel_rect = cancel_text.get_rect(center=self.btn_cancel_rect.center)
        Const.SCREEN.blit(cancel_text, cancel_rect)