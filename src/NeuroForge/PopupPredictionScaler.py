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
        self.training_data = self.model.TRI.training_data

    def header_text(self) -> str:
        return "Prediction Scaler Information"

    def content_to_display(self) -> List[List[str]]:
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
        params_col.append(self._format_params(target_scaler))

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
        params_col.append(self._format_params(target_scaler))

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

    def _format_params(self, scaler) -> str:
        """Format scaler parameters into a readable string."""
        if not scaler.params:
            return "-"

        params = scaler.params

        # Format based on common parameter types
        if 'median' in params and 'iqr' in params:
            return f"Med:{params['median']:.2f}, IQR:{params['iqr']:.2f}"
        elif 'min' in params and 'max' in params:
            return f"Min:{params['min']:.2f}, Max:{params['max']:.2f}"
        elif 'mean' in params and 'std' in params:
            return f"μ:{params['mean']:.2f}, σ:{params['std']:.2f}"
        elif 'max_abs' in params:
            return f"MaxAbs:{params['max_abs']:.2f}"
        elif 'offset' in params:
            return f"Offset:{params['offset']:.2f}"
        else:
            # Generic formatting for unknown param types
            return ", ".join(f"{k}:{v:.2f}" if isinstance(v, (int, float)) else f"{k}:{v}"
                             for k, v in params.items())

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