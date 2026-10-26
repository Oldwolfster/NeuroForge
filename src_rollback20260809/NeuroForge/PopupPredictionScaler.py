from typing import List
import json
from src.NeuroForge.Popup_Base import Popup_Base
from src.NeuroForge import Const


class PopupPredictionScaler(Popup_Base):
    """Popup showing prediction/target scaler information with current values."""

    def __init__(self, scaler_neuron):
        """
        Initialize popup for prediction scaler.

        Args:
            scaler_neuron: DisplayModel__NeuronScaler instance with is_input=False
        """
        super().__init__(
            text_color=Const.COLOR_BLACK,
            highlight_differences=False,
            column_width_overrides={0: 100,4:400}  # 👈 Fix column 0 width for labels
        )
        self.scaler_neuron = scaler_neuron
        self.model = scaler_neuron.model
        self.multi_scaler = self.model.config.scaler
        self.scaler_params = self.model.TRI.scaler_params
        self.training_data = self.model.TRI.training_data

    def header_text(self) -> str:
        return "Prediction Scaler Information"

    def content_to_displayOld(self) -> List[List[str]]:
        """Build columns showing prediction and target values with scaler info."""

        rs                  = Const.dm.get_sample_data(self.model.run_id)  # Get current iteration data
        prediction_raw      = rs.get("prediction_raw", "[]")  # Get prediction values (scaled and unscaled)
        prediction_unscaled = rs.get("prediction_unscaled", "[]")
        target_raw          = rs.get("target", "[]")     # Get target values (scaled and unscaled)
        target_unscaled     = rs.get("target_unscaled", "[]")

        # Parse JSON if needed
        try:
            pred_scaled = json.loads(prediction_raw) if isinstance(prediction_raw, str) else prediction_raw
            pred_unscaled = json.loads(prediction_unscaled) if isinstance(prediction_unscaled,
                                                                          str) else prediction_unscaled
            tgt_scaled = json.loads(target_raw) if isinstance(target_raw, str) else target_raw
            tgt_unscaled = json.loads(target_unscaled) if isinstance(target_unscaled, str) else target_unscaled
        except (json.JSONDecodeError, TypeError):
            pred_scaled = pred_unscaled = tgt_scaled = tgt_unscaled = []

        # Get the target scaler (last scaler in the list)
        target_scaler = self.multi_scaler.scalers[-1]

        # Get target feature name
        target_name = self.training_data.feature_labels[-1]

        # Build column data
        metric_col = [""]
        scaler_col = ["Scaler"]
        params_col = ["Parameters"]
        scaled_col = ["Scaled"]
        unscaled_col = ["Unscaled"]

        # Add prediction row
        metric_col.append("Prediction")
        scaler_col.append(target_scaler.name)
        params_col.append(target_scaler.describe_params(self.scaler_params[-1]))

        # Format prediction values
        if pred_unscaled:
            pred_unscaled_val = pred_unscaled[0] if isinstance(pred_unscaled, list) else pred_unscaled
            unscaled_col.append(f"{pred_unscaled_val:.3f}")
        else:
            unscaled_col.append("N/A")

        if pred_scaled:
            pred_scaled_val = pred_scaled[0] if isinstance(pred_scaled, list) else pred_scaled
            scaled_col.append(f"{pred_scaled_val:.3f}")
        else:
            scaled_col.append("N/A")

        # Add target row
        metric_col.append("Target")
        scaler_col.append(target_scaler.name)
        params_col.append(target_scaler.describe_params(self.scaler_params[-1]))

        # Format target values
        if tgt_unscaled:
            tgt_unscaled_val = tgt_unscaled[0] if isinstance(tgt_unscaled, list) else tgt_unscaled
            unscaled_col.append(f"{tgt_unscaled_val:.3f}")
        else:
            unscaled_col.append("N/A")

        if tgt_scaled:
            tgt_scaled_val = tgt_scaled[0] if isinstance(tgt_scaled, list) else tgt_scaled
            scaled_col.append(f"{tgt_scaled_val:.3f}")
        else:
            scaled_col.append("N/A")

        # Add error row
        metric_col.append("Error")
        scaler_col.append("")
        params_col.append("")

        # Calculate error (unscaled and scaled)
        try:
            if pred_unscaled and tgt_unscaled:
                pred_u = pred_unscaled[0] if isinstance(pred_unscaled, list) else pred_unscaled
                tgt_u = tgt_unscaled[0] if isinstance(tgt_unscaled, list) else tgt_unscaled
                error_unscaled = pred_u - tgt_u
                unscaled_col.append(f"{error_unscaled:.3f}")
            else:
                unscaled_col.append("N/A")

            if pred_scaled and tgt_scaled:
                pred_s = pred_scaled[0] if isinstance(pred_scaled, list) else pred_scaled
                tgt_s = tgt_scaled[0] if isinstance(tgt_scaled, list) else tgt_scaled
                error_scaled = pred_s - tgt_s
                scaled_col.append(f"{error_scaled:.3f}")
            else:
                scaled_col.append("N/A")
        except (TypeError, ValueError):
            unscaled_col.append("N/A")
            scaled_col.append("N/A")

        # Add description section
        self._add_scaler_description(metric_col, scaler_col, params_col, unscaled_col, scaled_col, target_scaler)

        return [metric_col, scaler_col, params_col, scaled_col, unscaled_col]

    def _add_scaler_description(self, metric_col, scaler_col, params_col, unscaled_col, scaled_col, scaler):
        """Add description rows for the target scaler."""

        # Add blank separator row
        metric_col.append("")
        scaler_col.append("")
        params_col.append("")
        unscaled_col.append("")
        scaled_col.append("")

        # Scaler name header
        metric_col.append(f"- {scaler.name} -")
        scaler_col.append("")
        params_col.append("")
        unscaled_col.append("")
        scaled_col.append("")

        # Description
        if scaler.desc:
            metric_col.append(scaler.desc)
            scaler_col.append("")
            params_col.append("")
            unscaled_col.append("")
            scaled_col.append("")

        # When to use
        if scaler.when_to_use:
            metric_col.append(f"Use: {scaler.when_to_use}")
            scaler_col.append("")
            params_col.append("")
            unscaled_col.append("")
            scaled_col.append("")

        # Best for
        if scaler.best_for:
            metric_col.append(f"Best: {scaler.best_for}")
            scaler_col.append("")
            params_col.append("")
            unscaled_col.append("")
            scaled_col.append("")

    def is_header_cell(self, col_index, row_index) -> bool:
        """First row is header."""
        return row_index == 0

    def draw_dividers(self, surf, col_widths):
        """Draw a line after the data rows and before descriptions."""
        import pygame

        # Find where the blank separator row is
        columns = self.content_to_display()
        if not columns:
            return

        data_rows = 0
        for i, cell in enumerate(columns[0]):
            if cell == "":
                data_rows = i
                break

        if data_rows > 0:
            # Draw horizontal line after data section
            y = self.y_coord_for_row(data_rows)
            x1 = Const.TOOLTIP_PADDING
            x2 = sum(col_widths) + Const.TOOLTIP_PADDING
            pygame.draw.line(surf, Const.COLOR_BLACK, (x1, y), (x2, y), 2)

    # ══════════════════════════════════════════════════════════════════════
    # update
    # ══════════════════════════════════════════════════════════════════════


    # PopupPredictionScaler.py  method: first  NEW: absent is None — zero is present
    def first(self, values):
        """The single value out of a list, a scalar, or nothing at all."""
        if values is None:              return None
        if isinstance(values, list):    return values[0] if values else None
        return values

    # PopupPredictionScaler.py  method: cell  NEW: one formatter, one guard
    def cell(self, values) -> str:
        val = self.first(values)
        return "N/A" if val is None else f"{val:.3f}"

    # PopupPredictionScaler.py  method: error_cell  NEW: N/A only when a side is truly missing
    def error_cell(self, pred, tgt) -> str:
        p, t = self.first(pred), self.first(tgt)
        return "N/A" if p is None or t is None else f"{p - t:.3f}"

    # PopupPredictionScaler.py  method: content_to_display  Update: helpers + neuron row order
    def content_to_display(self) -> List[List[str]]:
        """Build columns showing prediction and target values with scaler info."""

        rs = Const.dm.get_sample_data(self.model.run_id)  # Get current iteration data
        prediction_raw = rs.get("prediction_raw", "[]")  # Get prediction values (scaled and unscaled)
        prediction_unscaled = rs.get("prediction_unscaled", "[]")
        target_raw = rs.get("target", "[]")  # Get target values (scaled and unscaled)
        target_unscaled = rs.get("target_unscaled", "[]")

        # Parse JSON if needed
        try:
            pred_scaled = json.loads(prediction_raw) if isinstance(prediction_raw, str) else prediction_raw
            pred_unscaled = json.loads(prediction_unscaled) if isinstance(prediction_unscaled,
                                                                          str) else prediction_unscaled
            tgt_scaled = json.loads(target_raw) if isinstance(target_raw, str) else target_raw
            tgt_unscaled = json.loads(target_unscaled) if isinstance(target_unscaled, str) else target_unscaled
        except (json.JSONDecodeError, TypeError):
            pred_scaled = pred_unscaled = tgt_scaled = tgt_unscaled = []

        # Get the target scaler (last scaler in the list)
        target_scaler = self.multi_scaler.scalers[-1]

        # Get target feature name
        target_name = self.training_data.feature_labels[-1]
        params = target_scaler.describe_params(self.scaler_params[-1])

        # Build column data -- order matches the scaler neuron: Target, Prediction, Error
        metric_col = ["", "Target", "Prediction", "Error"]
        scaler_col = ["Scaler", target_scaler.name, target_scaler.name, ""]
        params_col = ["Parameters", params, params, ""]
        scaled_col = ["Scaled", self.cell(tgt_scaled), self.cell(pred_scaled),
                      self.error_cell(pred_scaled, tgt_scaled)]
        unscaled_col = ["Unscaled", self.cell(tgt_unscaled), self.cell(pred_unscaled),
                        self.error_cell(pred_unscaled, tgt_unscaled)]

        # Add description section
        self._add_scaler_description(metric_col, scaler_col, params_col, unscaled_col, scaled_col, target_scaler)

        return [metric_col, scaler_col, params_col, scaled_col, unscaled_col]