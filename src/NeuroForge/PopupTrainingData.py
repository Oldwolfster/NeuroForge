from typing import List

import pygame

from src.NeuroForge import Const
from src.NeuroForge.Popup_Base import Popup_Base
from src.NNA.engine.Config import Config



class PopupTrainingData(Popup_Base):
    def __init__(self):
        super().__init__(column_width_overrides={
                0: 20,
                2: 20,
            })




    def header_text(self):
        return "Arena(Training Data) Logic"


    def draw_dividers(self, surf, col_w):

        y = self.y_coord_for_row(1)
        #pygame.draw.line(surf, Const.COLOR_BLACK, (0,y),(ARCH_W,y),2)
    # no highlights required here
    def is_header_cell(self, col_index, row_index) -> bool:
        return False  #return col_index == 0 or row_index == 0

    def content_to_display(self):
        src = Const.TRIs[0].training_data.source_code
        lines = src.splitlines()

        # — extract the doc-string (without the """ markers) —
        doc_lines = []
        in_doc = False
        for line in lines:
            stripped = line.strip()
            # start of docstring
            if not in_doc and stripped.startswith('"""'):
                in_doc = True
                # grab anything after the opening """, e.g. on one-liner
                rest = stripped[3:]
                if rest:
                    doc_lines.append(rest)
                continue
            # end of docstring
            if in_doc and stripped.endswith('"""'):
                # grab anything before the closing """
                content = stripped[:-3].rstrip()
                if content:
                    doc_lines.append(content)
                in_doc = False
                break
            # middle of docstring
            if in_doc:
                doc_lines.append(stripped)

        # — now pull code after the generate_training_data signature —
        code_lines = []
        saw_def = False
        for line in lines:
            if not saw_def:
                if line.strip().startswith("def generate_training_data"):
                    saw_def = True
                continue
            # once we’ve seen the def, grab every subsequent line
            code_lines.append(line)

        # drop common indentation
        # find the smallest indent on non-blank code lines
        indents = [
            len(l) - len(l.lstrip())
            for l in code_lines if l.strip()
        ]
        min_indent = min(indents) if indents else 0
        code_lines = [l[min_indent:] for l in code_lines]

        d = len(doc_lines)
        c = len(code_lines)
        colCode     = [""] * (d + 1) + code_lines
        colComments = doc_lines     + [""] * (1 + c)
        colTail     = [" "] * (d + c+1)
        return [colCode, colComments, colTail]

        # build your two “columns” (first row empty for code, then code; first row docstring, then blanks)
        #colCode     = [""] + code_lines
        #colComments = ["\n".join(doc_lines)] + [""] * len(code_lines)
        #colComments = ["\n".join(doc_lines)] + [""] * 1

        #for _ in range(len(colComments)):
        #    colCode.insert(1,"")
        #return [colCode, colComments]
