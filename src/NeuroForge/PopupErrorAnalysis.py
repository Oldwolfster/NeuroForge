from typing import List

#from src.NNA.engine.UtilsPyGame import wrap_text
from src.NeuroForge.Popup_Base import Popup_Base
from src.NeuroForge import Const


class PopupErrorAnalysis(Popup_Base):
    """Popup explaining the Error Analysis Pass - the unsung hero of neural networks."""

    def __init__(self):
        super().__init__(
            column_width_overrides={0: 600}  # Single wide column for prose
        )

    def set_model_index(self, idx: int):
        """ADD THIS METHOD: Set which model's loss function to display"""
        self.model_index = idx
        # Force cache invalidation so it redraws with new model
        self.last_state = None

    def header_text(self):
        return "The Error Analysis Pass"

    def is_header_cell(self, col_index, row_index) -> bool:
        return False  # No cell headers, just flowing text

    def get_loss_function_details(self) -> List[str]:
        """Extract loss function details from the current model"""

        # self.model_index = 0 #TODO FIX THIS!!!

        if self.model_index >= len(Const.TRIs):
            return []

        loss_fn = Const.TRIs[self.model_index].config.loss_function

        details = [
            f"============ LOSS FUNCTION =============",
            f" {loss_fn.name} ({loss_fn.short_name})",
            f"========================================",
            "",
        ]

        if loss_fn.desc:
            details.extend(["WHAT IT IS:"])
            details.extend(wrap_text(loss_fn.desc, 80))
            details.extend([""])

        if loss_fn.desc:
            details.extend(["WHEN TO USE:"])
            details.extend(wrap_text(loss_fn.when_to_use, 80))
            details.extend([""])

        if loss_fn.desc:
            details.extend(["BEST FOR:"])
            details.extend(wrap_text(loss_fn.best_for, 80))
            details.extend([""])




        if loss_fn.derivative_formula:
            details.extend([
                f"GRADIENT FORMULA:",
                f"  {loss_fn.derivative_formula}",
                "",
            ])

        details.append("=" * 80)
        details.append("")

        return details

    def content_to_display(self) -> List[List[str]]:
        """Explain the three-phase cycle and what each display box means."""

        # Start with loss function details
        content = self.get_loss_function_details()

        # Then add the existing explanation
        content.extend([

            "Neural Networks 'brute force' a 3 step process.",
            "",
            "--- THE THREE-STEP CYCLE ---",
            "",
            "1)  Guess (Forward Pass)",
            "      Network makes its best guess",
            "",
            "2)  Check it (Error Analysis Pass) ← YOU ARE HERE",
            "    Compare prediction to actual target",
            "    Which way can we nudge to improve it next time?",
            "",
            "3)  ADJUST (Backward Pass)",
            "      Give it a tiny nudge that way",
            "",
            "Repeat until it can learn no more... That's machine learning.",
            "",
            "--- WHAT EACH BOX IN THIS PANEL MEANS ---",
            "",
            "TARGET: What the answer SHOULD be (from your training data)",
            "",
            "PREDICTION: What your network ACTUALLY predicted",
            "",
            "ERROR: Target minus Prediction (positive = too low, negative = too high)",
            "",
            "MSE GRADIENT: The 'blame signal' that starts the backward pass",
            "    (Notice the arrow pointing left? That's where backprop begins)",
            "",
            "NNs work because of REPETITION, not precision. Each adjustment is tiny.",
            "But do it enough times, aiming in the right direction? Magic happens.",

            '''
                ═══ THE REAL TALK ABOUT LOSS ═══    
                Loss functions are both CRITICAL and DANGEROUS:
            
                Critical: You MUST pick one.  It's the tool you have to tweak weights. Pick wrong, your model never learns. Pick right, it works.
            
                Dangerous: The path from error to gradient is full of misleading names.  The
                math geeks named these things, and we computer nerds trusted them to be clear.  OOPS!
            
                The bigger danger:  Tiny changes to your config can have huge impact on performance and which settings do the
                        if best:
            
                This makes textbook rules, have limited value.
            
                That loss function that failed with 3 neurons? Might dominate with 4.
            
                What actually works: Cast a wide net.  Find what works for yu
            
                for YOUR specific scenario.Don't make rules - make measurements.
                One
                more
                thing:
            
                Loss
                Value(the
                number): Just
                a
                scoreboard
                Loss
                Gradient(at
                the
                bottom): This is what
                actually
                trains
                your
                model
            
            
                '''
        ])

        return [content]