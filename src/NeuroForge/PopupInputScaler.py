from typing import List
import json
from src.NeuroForge.Popup_Base import Popup_Base
from src.NeuroForge import Const


class PopupInputScaler(Popup_Base):
    """Popup showing input scaler information with parameters and current values."""

    def __init__(self, scaler_neuron):
        """
        Initialize popup for input scaler.

        Args:
            scaler_neuron: DisplayModel__NeuronScaler instance with is_input=True
        """
        super().__init__(
            text_color=Const.COLOR_BLACK,
            highlight_differences=False,
            column_width_overrides={0: 100,5:400}  # 👈 Fix column 0 width for labels
        )
        self.scaler_neuron = scaler_neuron
        self.model = scaler_neuron.model
        self.multi_scaler = self.model.config.scaler
        self.training_data = self.model.TRI.training_data

    def header_text(self) -> str:
        return "Input Scaler Information"

    def content_to_display(self) -> List[List[str]]:
        """Build columns showing feature, scaler, params, unscaled, and scaled values."""

        # Get current iteration data
        rs = Const.dm.get_model_iteration_data(self.model.run_id)

        # Get unscaled inputs
        raw_inputs = rs.get("inputs_unscaled", "[]")
        try:
            inputs_unscaled = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
        except json.JSONDecodeError:
            inputs_unscaled = []

        # Get scaled inputs from the neuron visualizer
        if hasattr(self.scaler_neuron.neuron_visualizer, 'scaled_inputs'):
            inputs_scaled = self.scaler_neuron.neuron_visualizer.scaled_inputs
        else:
            inputs_scaled = []

        # Build column data
        feature_col = ["Feature"]
        scaler_col = ["Scaler"]
        params_col = ["Parameters"]
        unscaled_col = ["Unscaled"]
        scaled_col = ["Scaled"]

        # Get feature labels (exclude target)
        feature_labels = self.training_data.feature_labels[:-1]

        # 🔍 DEBUG: Let's see what we're working with
        """
        print("\n" + "=" * 60)
        print("🔍 POPUP INPUT SCALER DEBUG")
        print("=" * 60)
        print(f"training_data.feature_labels (full): {self.training_data.feature_labels}")
        print(f"feature_labels (sliced): {feature_labels}")
        print(f"Number of feature_labels: {len(feature_labels)}")
        print(f"Number of scalers: {len(self.multi_scaler.scalers)}")
        print(f"Number of input scalers (excluding target): {len(self.multi_scaler.scalers) - 1}")
        print(f"\nScaler instances (are they the same object?):")
        for i in range(len(self.multi_scaler.scalers) - 1):  # Exclude target
            scaler = self.multi_scaler.scalers[i]
            print(f"  [{i}] {id(scaler):16x} - {scaler.name} - params: {scaler.params}")
        print("=" * 60 + "\n")
        """

        # Populate rows for each input feature
        for i, label in enumerate(feature_labels):
            scaler = self.multi_scaler.scalers[i]

            # Feature name
            feature_col.append(label)

            # Scaler name
            scaler_col.append(scaler.name)

            # Parameters - format based on scaler type
            params_str = self._format_params(scaler)
            params_col.append(params_str)

            # Unscaled value
            if i < len(inputs_unscaled):
                unscaled_col.append(f"{inputs_unscaled[i]:.3f}")
            else:
                unscaled_col.append("N/A")

            # Scaled value
            if i < len(inputs_scaled):
                scaled_col.append(f"{inputs_scaled[i]:.3f}")
            else:
                scaled_col.append("N/A")

        # Add description section for unique scalers
        self._add_scaler_descriptions(feature_col, scaler_col, params_col, unscaled_col, scaled_col)

        return [feature_col, scaler_col, params_col, unscaled_col, scaled_col]

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

    def _add_scaler_descriptions(self, feature_col, scaler_col, params_col, unscaled_col, scaled_col):
        """Add description rows for each unique scaler type used."""

        # Find unique scalers
        unique_scalers = {}
        for scaler in self.multi_scaler.scalers[:-1]:  # Exclude target scaler
            if scaler.name not in unique_scalers:
                unique_scalers[scaler.name] = scaler

        # Add blank separator row
        feature_col.append("")
        scaler_col.append("")
        params_col.append("")
        unscaled_col.append("")
        scaled_col.append("")

        # Add description for each unique scaler
        for scaler_name, scaler in unique_scalers.items():
            # Scaler name header
            feature_col.append(f"- {scaler_name} -")
            scaler_col.append("")
            params_col.append("")
            unscaled_col.append("")
            scaled_col.append("")

            # Description
            if scaler.desc:
                feature_col.append(scaler.desc)
                scaler_col.append("")
                params_col.append("")
                unscaled_col.append("")
                scaled_col.append("")

            # When to use
            if scaler.when_to_use:
                feature_col.append(f"Use: {scaler.when_to_use}")
                scaler_col.append("")
                params_col.append("")
                unscaled_col.append("")
                scaled_col.append("")

            # Best for
            if scaler.best_for:
                feature_col.append(f"Best: {scaler.best_for}")
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

        # Find where the blank separator row is (after data, before descriptions)
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