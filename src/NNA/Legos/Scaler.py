from src.NNA.engine.TrainingData import TrainingData

class MultiScaler:
    def __init__(self, training_data: TrainingData):
        #print("initializing Multiscaler")
        #if not training_data: return
        self.training_data = training_data
        #print("training_dat")
        self.training_data.verify_targets_unchanged("Multiscaler constructor")

        self.scalers = [Scaler_NONE for _ in range(training_data.input_count + 1)]  # +1 for target
        self.scaled_data = []
        self.scaled_samples = []  # ← Packed [inputs..., target]
        self.unscaled_samples = training_data.get_list()
        self.mean_target = None

    def set_input_scaler(self, scaler, index):
        if index == len(self.scalers) - 1:
            print(f"[Warning] Attempted to set scaler on target column (index {index}). Ignored.")
            return
        self.scalers[index] = scaler.clone()  # ✅ Create independent instance


    def set_target_scaler(self, scaler):
        """
        Assigns a scaler to the target (last column).
        """
        self.scalers[-1] = scaler.clone()

    def set_all_input_scalers(self, scaler):
        """
        Apply `scaler` to every input feature,
        leaving the last entry (the target) untouched.
        """
        for i in range(self.training_data.input_count):
            self.scalers[i] = scaler.clone()  # ✅ Create independent instance

    def set_target_scaler(self, scaler):
        self.scalers[-1] = scaler.clone()

    def scale_all(self):
        """
        Scales training data using the assigned scalers. Populates scaled_samples and returns inputs + targets.
        """
        raw_samples = self.unscaled_samples
        transposed = list(zip(*raw_samples))
        self.scaled_data = [
            scaler.scale(list(feature)) for scaler, feature in zip(self.scalers, transposed)
        ]
        # Split back into inputs and targets
        inputs = list(zip(*self.scaled_data[:-1]))  # all but last column
        targets = self.scaled_data[-1]
        self.scaled_samples = [list(inp) + [target] for inp, target in zip(inputs, targets)]

        # compute and store mean of the *scaled* targets
        self.mean_target = sum(targets) / len(targets)

        #print(f"raw    data\t{raw_samples[:4]}")
        #print(f"scaled data\t{self.scaled_data}")
        #print(f"sample data\t{self.scaled_samples}")

        return inputs, targets

    def unscale_input(self, input_index: int, value):
        scaler = self.scalers[input_index]
        return scaler.unscale(scaler.params, value)

    def unscale_target(self, value):
        scaler = self.scalers[-1]
        return scaler.unscale(value)

    @property
    def inputs_are_scaled(self):
        return not all(scaler is Scaler_NONE for scaler in self.scalers[:-1])

    @property
    def target_is_scaled(self):
        return not self.scalers[-1] is Scaler_NONE

    @property
    def all_unscaled(self):
        return self.inputs_are_unscaled and self.target_is_unscaled

    def get_scaling_labels(self):
        labels = list(self.training_data.feature_labels)
        #labels.append("Target")  # Prevent extra line mismatch
        return [f"scaling: {label}" for label in labels]

    def get_scaling_names(self):
        #return  [scaler.name for scaler in self.scalers[:-1]] + ["Coming Soon"]  # blank for placeholder
        return  [scaler.name for scaler in self.scalers]    # blank for placeholder

    def __repr__(self):
        return f"MultiScaler({[s.name for s in self.scalers]})"


class Scaler:#TODO rename StrategyScaler
    def __init__(self, method, name="Scaler", desc="", when_to_use="", best_for=""):
        self.method = method
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for
        self.params = {}
        self.scaled_data = []  # ← store scaled inputs for inspection

    def scale(self, values):
        """
        Fits the scaler to the flat list of values and returns the scaled result.
        Also stores internal params and scaled data.
        """
        self.params.clear()
        self.scaled_data = self.method.scale(self.params, values)
        return self.scaled_data

    def unscale(self, scaled):
        """
        Un-scales a single value or list of values back to the original space.
        """
        return self.method.unscale(self.params, scaled)

    def __repr__(self):
        return f"Scaler(name={self.name})"


    def clone(self):
        """Create a new instance with the same method and metadata but independent params."""
        return Scaler(
            method=self.method,  # Methods are stateless, can be shared
            name=self.name,
            desc=self.desc,
            when_to_use=self.when_to_use,
            best_for=self.best_for
        )


class MinMaxMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1

        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            min_val = min(vals)
            max_val = max(vals)
            range_val = max_val - min_val
            params['min'] = min_val
            params['max'] = max_val

            if range_val == 0:
                return [0.0 for _ in vals]  # or 0.5 if you prefer midpoint
            return [(v - min_val) / range_val for v in vals]

        else:
            min_vals = [min(v[i] for v in values) for i in range(dim)]
            max_vals = [max(v[i] for v in values) for i in range(dim)]
            params['min'] = min_vals
            params['max'] = max_vals

            scaled = []
            for v in values:
                row = []
                for i in range(dim):
                    range_val = max_vals[i] - min_vals[i]
                    if range_val == 0:
                        row.append(0.0)  # or 0.5
                    else:
                        row.append((v[i] - min_vals[i]) / range_val)
                scaled.append(row)
            return scaled


    def unscale(self, params, x):
        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [
                [xv[i] * (params['max'][i] - params['min'][i]) + params['min'][i] for i in range(len(xv))]
                for xv in x
            ]
        elif isinstance(x, list):
            return [
                xv * (params['max'] - params['min']) + params['min'] for xv in x
            ]
        else:
            return x * (params['max'] - params['min']) + params['min']


Scaler_MinMax = Scaler(
    method=MinMaxMethod(),
    name="Min-Max",
    desc="Scales data to a fixed range [0,1] using min and max.",
    when_to_use="Targets or inputs with known range and no severe outliers.",
    best_for="Tanh, Sigmoid, Linear and the kitchen sink."
)
class NoScalingMethod:
    def scale(self, params, values):
        return values  # Pass-through

    def unscale(self, params, x):
        return x  # Pass-through


Scaler_NONE = Scaler(
    method=NoScalingMethod(),
    name="No Scaling",
    desc="Pass-through scaler that performs no transformation.",
    when_to_use="Use when raw values are already appropriately scaled or scaling is undesirable.",
    best_for="Debugging, baselines, or models insensitive to scale."
)

import math

class LogMinMaxMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1
        epsilon = 1e-6

        def safe_log(x):
            return math.log(max(x, epsilon))

        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            log_vals = [safe_log(v) for v in vals]
            min_val = min(log_vals)
            max_val = max(log_vals)
            range_val = max_val - min_val

            params['min'] = min_val
            params['max'] = max_val
            params['log'] = True  # for clarity

            if range_val == 0:
                return [0.0 for _ in vals]
            return [(lv - min_val) / range_val for lv in log_vals]

        else:
            # Multi-dim version
            log_vals = [
                [safe_log(v[i]) for i in range(dim)] for v in values
            ]
            min_vals = [min(v[i] for v in log_vals) for i in range(dim)]
            max_vals = [max(v[i] for v in log_vals) for i in range(dim)]

            params['min'] = min_vals
            params['max'] = max_vals
            params['log'] = True

            scaled = []
            for lv in log_vals:
                row = []
                for i in range(dim):
                    range_val = max_vals[i] - min_vals[i]
                    if range_val == 0:
                        row.append(0.0)
                    else:
                        row.append((lv[i] - min_vals[i]) / range_val)
                scaled.append(row)
            return scaled

    def unscale(self, params, x):
        def safe_exp(x):
            return math.exp(x)

        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [
                [safe_exp(xv[i] * (params['max'][i] - params['min'][i]) + params['min'][i]) for i in range(len(xv))]
                for xv in x
            ]
        elif isinstance(x, list):
            return [
                safe_exp(xv * (params['max'] - params['min']) + params['min']) for xv in x
            ]
        else:
            return safe_exp(x * (params['max'] - params['min']) + params['min'])


Scaler_LogMinMax = Scaler(
    method=LogMinMaxMethod(),
    name="Log-MinMax",
    desc="Applies log transform followed by Min-Max scaling. Handles skew and outliers.",
    when_to_use="When features have extreme outliers or long-tail distributions (e.g. price, fare, salary).",
    best_for="Features like Fare that bully others due to scale, even after normalization."
)


class ZScoreMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1
        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
            params['mean'] = mean
            params['std'] = std
            return [(v - mean) / std for v in vals]
        else:
            means = [sum(v[i] for v in values) / len(values) for i in range(dim)]
            stds = [
                (sum((v[i] - means[i]) ** 2 for v in values) / len(values)) ** 0.5 or 1.0
                for i in range(dim)
            ]
            params['mean'] = means
            params['std'] = stds
            return [
                [(v[i] - means[i]) / stds[i] for i in range(dim)] for v in values
            ]

    def unscale(self, params, x):
        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [
                [xv[i] * params['std'][i] + params['mean'][i] for i in range(len(xv))]
                for xv in x
            ]
        elif isinstance(x, list):
            return [xv * params['std'] + params['mean'] for xv in x]
        else:
            return x * params['std'] + params['mean']


Scaler_ZScore = Scaler(
    method=ZScoreMethod(),
    name="Z-Score",
    desc="Scales data to zero mean and unit variance using standard score.",
    when_to_use="Data with unknown range or needing normalization for gradient stability.",
    best_for="Most ML algorithms, especially gradient-based methods."
)


class MinMaxZeroCenteredMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1

        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            min_val = min(vals)
            max_val = max(vals)
            range_val = max_val - min_val
            params['min'] = min_val
            params['max'] = max_val

            if range_val == 0:
                return [0.0 for _ in vals]
            return [((v - min_val) / range_val) * 2 - 1 for v in vals]

        else:
            min_vals = [min(v[i] for v in values) for i in range(dim)]
            max_vals = [max(v[i] for v in values) for i in range(dim)]
            params['min'] = min_vals
            params['max'] = max_vals

            scaled = []
            for v in values:
                row = []
                for i in range(dim):
                    range_val = max_vals[i] - min_vals[i]
                    if range_val == 0:
                        row.append(0.0)
                    else:
                        row.append(((v[i] - min_vals[i]) / range_val) * 2 - 1)
                scaled.append(row)
            return scaled

    def unscale(self, params, x):
        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [
                [((xv[i] + 1) / 2) * (params['max'][i] - params['min'][i]) + params['min'][i] for i in range(len(xv))]
                for xv in x
            ]
        elif isinstance(x, list):
            return [
                ((xv + 1) / 2) * (params['max'] - params['min']) + params['min'] for xv in x
            ]
        else:
            return ((x + 1) / 2) * (params['max'] - params['min']) + params['min']


Scaler_MinMax_Neg1to1 = Scaler(
    method=MinMaxZeroCenteredMethod(),
    name="Min-Max-Zero-Centered",
    desc="Scales data to the range [-1,1] using min and max.",
    when_to_use="When using tanh or other symmetric activation functions.",
    best_for="Output targets or inputs feeding into Tanh activations."
)


class MaxAbsMethod:
    def scale(self, params, values):
        dim = len(values[0]) if isinstance(values[0], (list, tuple)) else 1
        if dim == 1:
            vals = [v if isinstance(v, (int, float)) else v[0] for v in values]
            max_abs = max(abs(v) for v in vals) or 1.0
            params['max_abs'] = max_abs
            return [v / max_abs for v in vals]
        else:
            max_abs = [max(abs(v[i]) for v in values) or 1.0 for i in range(dim)]
            params['max_abs'] = max_abs
            return [[v[i] / max_abs[i] for i in range(dim)] for v in values]

    def unscale(self, params, x):
        if isinstance(x, list) and isinstance(x[0], (list, tuple)):
            return [[xv[i] * params['max_abs'][i] for i in range(len(xv))] for xv in x]
        elif isinstance(x, list):
            return [xv * params['max_abs'] for xv in x]
        else:
            return x * params['max_abs']


Scaler_MaxAbs = Scaler(
    method=MaxAbsMethod(),
    name="Max-Abs",
    desc="Scales data to [-1, 1] range using the maximum absolute value.",
    when_to_use="Inputs centered around zero or signed data without strong outliers.",
    best_for="Linear models or cases where sign preservation is important."
)

class RobustScalerMethod:

    def _sanitize_iqr(self, iqr, values=None):
        """
        Ensure that no dimension of `iqr` is zero.
        If IQR is 0 but data has variation, use standard deviation instead.
        """
        import numpy as np

        if np.isscalar(iqr):
            if iqr == 0 and values is not None:
                # Check if data actually has variation
                std_val = np.std(values)
                return std_val if std_val > 0 else 1.0
            return 1.0 if iqr == 0 else iqr
        else:
            iqr = np.array(iqr)
            if values is not None:
                std_vals = np.std(values, axis=0)
                for i in range(len(iqr)):
                    if iqr[i] == 0 and std_vals[i] > 0:
                        iqr[i] = std_vals[i]
            iqr[iqr == 0] = 1.0
            return iqr

    def _sanitize_iqrOld(self, iqr):
        """
        Ensure that no dimension of `iqr` is zero.
        Works whether `iqr` is a scalar or an array.
        """
        import numpy as np

        if np.isscalar(iqr):
            return 1.0 if iqr == 0 else iqr
        else:
            # make sure it’s an array
            iqr = np.array(iqr)
            iqr[iqr == 0] = 1.0
            return iqr

    def scale(self, params, values):
        import numpy as np
        arr = np.array(values)

        # Compute median and (raw) IQR
        params['median'] = np.median(arr, axis=0)
        raw_iqr         = np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)

        # Sanitize: avoid division by zero
        params['iqr'] = self._sanitize_iqr(raw_iqr)

        # Do the scaling
        return ((arr - params['median']) / params['iqr']).tolist()

    def unscale(self, params, x):
        import numpy as np
        x = np.array(x)
        return (x * params['iqr'] + params['median']).tolist()


Scaler_Robust = Scaler(
    method=RobustScalerMethod(),
    name="Robust",
    desc="Uses median and IQR to reduce the influence of outliers.",
    when_to_use="Data with heavy outliers or skewed distributions.",
    best_for="Linear models or datasets with rare extreme values."
)


class LogScalerMethod:
    def scale(self, params, values):
        import numpy as np
        arr = np.array(values)
        params['offset'] = np.abs(np.min(arr)) + 1 if np.min(arr) <= 0 else 0
        return np.log(arr + params['offset']).tolist()

    def unscale(self, params, x):
        import numpy as np
        return (np.exp(x) - params['offset']).tolist()


Scaler_Log = Scaler(
    method=LogScalerMethod(),
    name="Log",
    desc="Applies natural log transform to reduce skew.",
    when_to_use="Data with exponential growth or multiplicative relationships.",
    best_for="Financial, biological, or power-law data."
)