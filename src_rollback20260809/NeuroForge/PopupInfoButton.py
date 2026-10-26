from typing import List

import pygame

from src.NeuroForge import Const
from src.NeuroForge.Popup_Base import Popup_Base
from src.NNA.engine.Config import Config



class PopupInfoButton(Popup_Base):
    def __init__(self):
        super().__init__(column_width_overrides={
            })

    def header_text(self):
        return "Heads Up: We Do Things Differently Around Here!"


    def draw_dividers(self, surf, col_w):

        y = self.y_coord_for_row(1)
        #pygame.draw.line(surf, Const.COLOR_BLACK, (0,y),(ARCH_W,y),2)
    # no highlights required here
    def is_header_cell(self, col_index, row_index) -> bool:
        return   row_index == 0     # col_index == 0

    def _split_and_clean(self, text: str) -> List[str]:
        # One small job: turn a multiline string into a list of non-empty lines
        return [line.strip() for line in text.splitlines() if line.strip()]


    def content_to_display(self) -> List[List[str]]:
        def raw_text_verbiage() -> str:
            return """
            Welcome to NeuroForge—our visuals and terminology aren’t the “textbook” defaults, so here’s what to expect:
        
            Neurons as Machines
            Each neuron is a little machine that contains its own weights, bias, activations; most importantly, inputs and output.
        
            Arrows = Outputs
            The arrows between neurons show the output traveling from one neuron to the input on the next, not the “weight” itself.
        
            Language Should Clarify — Not Confuse
            We’ve found that the traditional terminology often obscures more than it reveals. For example:
        
            “Linear Activation Function” sounds like it does something, but it really just means: no activation at all.
        
            The word “Gradient” is dangerous.  
            It shows up everywhere, but its meaning shifts constantly.  (Some story about a 17D mountain LOL)
            Most of the time, it just means:  
            “How much should this weight change?”  
            We skip the jargon and say exactly that.
            
            For regression tasks, "Accuracy" is defined as 1 - (MAE / mean target), 
            providing an intuitive % that reflects prediction closeness.

            
            We aim for:
            * Simple but precise language
            * Visuals that match intuition
            * Transparency over tradition
        
            If you're used to academic ML tools, this might feel a little unorthodox — and that's the point.
        
            Thanks for exploring NeuroForge. We hope it helps make the ideas click! :)
            """

        # grab the raw text
        src = raw_text_verbiage()
        # split into lines, strip whitespace, and drop any empty lines
        lines = [
            line.strip()                    # removes leading spaces (so your markdown bullets line up)
            for line in src.splitlines()    # breaks the docstring into individual lines
            #if line.strip()                 # skips any blank lines entirely
        ]
        # one column only
        return [lines]

