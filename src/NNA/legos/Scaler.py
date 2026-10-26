# Scaler.py
import math


def flatten_column(values):
    """Strategies receive ONE transposed column. Normalize stray 1-tuples to scalars."""
    return [v if isinstance(v, (int, float)) else v[0] for v in values]


class Scaler:
    """
    Immutable strategy. Holds NO params and NO data -- both live on the TRI.
    Shared singleton per method; never cloned, never mutated.
        fit       -> writes params.  The ONLY thing that may look at data to learn.
        transform -> reads params.   Safe for validate and test.
        unscale   -> reads params.
    """
    by_name = {}    # name -> Scaler singleton, populated as each is constructed

    def __init__(self, method, name="Scaler", desc="", when_to_use="", best_for=""):
        self.method         = method
        self.name           = name
        self.desc           = desc
        self.when_to_use    = when_to_use
        self.best_for       = best_for
        Scaler.by_name[name] = self

    def fit(self, params, values):
        self.method.fit         (params, values)

    def transform(self, params, values):
        return self.method.transform  (params, values)

    def unscale(self, params, scaled):
        return self.method.unscale    (params, scaled)

    def describe_params(self, params):
        if not params:  return "-"
        return self.method.describe_params(params)

    def __repr__(self):
        return f"Scaler(name={self.name})"


class MultiScaler:
    """
    Column-to-strategy assignment. Stateless beyond that assignment.
    All params are supplied by the caller (the TRI that owns them).
    """
    def __init__(self, input_count): #, feature_labels):
        self.input_count    = input_count
        #self.feature_labels = feature_labels
        self.scalers        = [Scaler_NONE for _ in range(input_count + 1)]   # +1 for target

    def set_input_scaler(self, scaler, index):
        if index == len(self.scalers) - 1:
            print(f"[Warning] Attempted to set scaler on target column (index {index}). Ignored.")
            return
        self.scalers[index] = scaler

    def set_all_input_scalers(self, scaler):
        for i in range(self.input_count):
            self.scalers[i] = scaler

    def set_target_scaler(self, scaler):
        self.scalers[-1] = scaler

    def columns_of(self, samples):
        """Rows of [inputs..., target] -> one column per scaler."""
        return list(zip(*samples))

    def fit_all(self, params_list, samples):
        """Learn from TRAINING samples only. Called once per run."""
        for scaler, params, column in zip(self.scalers, params_list, self.columns_of(samples)):
            scaler.fit(params, list(column))

    def transform_all(self, params_list, samples):
        """Apply frozen params to any phase's samples. Returns packed [inputs..., target]."""
        scaled_cols = [scaler.transform(params, list(column))
                       for scaler, params, column in zip(self.scalers, params_list, self.columns_of(samples))]
        inputs      = list(zip(*scaled_cols[:-1]))
        targets     = scaled_cols[-1]
        return [list(inp) + [target] for inp, target in zip(inputs, targets)]

    def unscale_input(self, params_list, input_index, value):
        return self.scalers[input_index].unscale(params_list[input_index], value)

    def unscale_target(self, params_list, value):
        return self.scalers[-1].unscale(params_list[-1], value)

    @property
    def inputs_are_scaled(self):
        return not all(scaler is Scaler_NONE for scaler in self.scalers[:-1])

    @property
    def target_is_scaled(self):
        return not self.scalers[-1] is Scaler_NONE

    #def get_scaling_labels(self):        return [f"scaling: {label}" for label in self.feature_labels]

    def get_scaling_names(self):
        return [scaler.name for scaler in self.scalers]

    def __repr__(self):
        return f"MultiScaler({[s.name for s in self.scalers]})"


class NoScalingMethod:
    def fit(self, params, values):
        pass                                            # nothing to learn

    def transform(self, params, values):
        return flatten_column(values)

    def unscale(self, params, x):
        return x
    def describe_params(self, params):        # NoScalingMethod
        return "-"


class MinMaxMethod:
    def fit(self, params, values):
        vals            = flatten_column(values)
        params['min']   = min(vals)
        params['max']   = max(vals)

    def transform(self, params, values):
        vals            = flatten_column(values)
        range_val       = params['max'] - params['min']
        if range_val    == 0:   return [0.0 for _ in vals]
        return [(v - params['min']) / range_val for v in vals]

    def unscale(self, params, x):
        span            = params['max'] - params['min']
        if isinstance(x, list): return [v * span + params['min'] for v in x]
        return x * span + params['min']
    def describe_params(self, params):        # MinMaxMethod
        return f"Min:{params['min']:.2f}, Max:{params['max']:.2f}"


class MinMaxZeroCenteredMethod:
    def fit(self, params, values):
        vals            = flatten_column(values)
        params['min']   = min(vals)
        params['max']   = max(vals)

    def transform(self, params, values):
        vals            = flatten_column(values)
        range_val       = params['max'] - params['min']
        if range_val    == 0:   return [0.0 for _ in vals]
        return [((v - params['min']) / range_val) * 2 - 1 for v in vals]

    def unscale(self, params, x):
        span            = params['max'] - params['min']
        if isinstance(x, list): return [((v + 1) / 2) * span + params['min'] for v in x]
        return ((x + 1) / 2) * span + params['min']
    def describe_params(self, params):        # MinMaxZeroCenteredMethod
        return f"Min:{params['min']:.2f}, Max:{params['max']:.2f} -> [-1,1]"


class ZScoreMethod:
    def fit(self, params, values):
        vals            = flatten_column(values)
        mean            = sum(vals) / len(vals)
        params['mean']  = mean
        params['std']   = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0

    def transform(self, params, values):
        vals            = flatten_column(values)
        return [(v - params['mean']) / params['std'] for v in vals]

    def unscale(self, params, x):
        if isinstance(x, list): return [v * params['std'] + params['mean'] for v in x]
        return x * params['std'] + params['mean']
    def describe_params(self, params):        # ZScoreMethod
        return f"μ:{params['mean']:.2f}, σ:{params['std']:.2f}"

class MaxAbsMethod:
    def fit(self, params, values):
        vals                = flatten_column(values)
        params['max_abs']   = max(abs(v) for v in vals) or 1.0

    def transform(self, params, values):
        vals                = flatten_column(values)
        return [v / params['max_abs'] for v in vals]

    def unscale(self, params, x):
        if isinstance(x, list): return [v * params['max_abs'] for v in x]
        return x * params['max_abs']

    def describe_params(self, params):        # MaxAbsMethod
        return f"MaxAbs:{params['max_abs']:.2f}"


class RobustScalerMethod:
    def fit(self, params, values):
        vals                = flatten_column(values)
        sorted_vals         = sorted(vals)
        params['median']    = self.median_of(sorted_vals)
        raw_iqr             = self.percentile(sorted_vals, 75) - self.percentile(sorted_vals, 25)
        params['iqr']       = self.sanitize_iqr(raw_iqr, vals)

    def transform(self, params, values):
        vals                = flatten_column(values)
        return [(v - params['median']) / params['iqr'] for v in vals]

    def unscale(self, params, x):
        if isinstance(x, list): return [v * params['iqr'] + params['median'] for v in x]
        return x * params['iqr'] + params['median']

    def median_of(self, sorted_vals):
        n = len(sorted_vals)
        if n % 2 == 1:  return sorted_vals[n // 2]
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    def percentile(self, sorted_vals, p):
        n = len(sorted_vals)
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    def sanitize_iqr(self, iqr, values):
        if iqr != 0:    return iqr
        mean        = sum(values) / len(values)
        variance    = sum((v - mean) ** 2 for v in values) / len(values)
        std         = variance ** 0.5
        return std if std > 0 else 1.0

    def describe_params(self, params):        # RobustScalerMethod
        return f"Med:{params['median']:.2f}, IQR:{params['iqr']:.2f}"

class LogMinMaxMethod:
    def fit(self, params, values):
        log_vals        = [self.safe_log(v) for v in flatten_column(values)]
        params['min']   = min(log_vals)
        params['max']   = max(log_vals)
        params['log']   = True

    def transform(self, params, values):
        log_vals        = [self.safe_log(v) for v in flatten_column(values)]
        range_val       = params['max'] - params['min']
        if range_val    == 0:   return [0.0 for _ in log_vals]
        return [(lv - params['min']) / range_val for lv in log_vals]

    def unscale(self, params, x):
        span            = params['max'] - params['min']
        if isinstance(x, list): return [math.exp(v * span + params['min']) for v in x]
        return math.exp(x * span + params['min'])

    def safe_log(self, x):
        return math.log(max(x, 1e-6))

    def describe_params(self, params):        # LogMinMaxMethod
        return f"log then Min:{params['min']:.2f}, Max:{params['max']:.2f}"

Scaler_NONE = Scaler(
    method          = NoScalingMethod(),
    name            = "No Scaling",
    desc            = "Pass-through scaler that performs no transformation.",
    when_to_use     = "Use when raw values are already appropriately scaled or scaling is undesirable.",
    best_for        = "Debugging, baselines, or models insensitive to scale."
)

Scaler_MinMax = Scaler(
    method          = MinMaxMethod(),
    name            = "Min-Max",
    desc            = "Scales data to a fixed range [0,1] using min and max.",
    when_to_use     = "Targets or inputs with known range and no severe outliers.",
    best_for        = "Tanh, Sigmoid, Linear and the kitchen sink."
)

Scaler_MinMax_Neg1to1 = Scaler(
    method          = MinMaxZeroCenteredMethod(),
    name            = "MM-Zero-Cent",
    desc            = "Scales data to the range [-1,1] using min and max.",
    when_to_use     = "When using tanh or other symmetric activation functions.",
    best_for        = "Output targets or inputs feeding into Tanh activations."
)

Scaler_ZScore = Scaler(
    method          = ZScoreMethod(),
    name            = "Z-Score",
    desc            = "Scales data to zero mean and unit variance using standard score.",
    when_to_use     = "Data with unknown range or needing normalization for gradient stability.",
    best_for        = "Most ML algorithms, especially gradient-based methods."
)

Scaler_MaxAbs = Scaler(
    method          = MaxAbsMethod(),
    name            = "Max-Abs",
    desc            = "Scales data to [-1, 1] range using the maximum absolute value.",
    when_to_use     = "Inputs centered around zero or signed data without strong outliers.",
    best_for        = "Linear models or cases where sign preservation is important."
)

Scaler_Robust = Scaler(
    method          = RobustScalerMethod(),
    name            = "Robust",
    desc            = "Uses median and IQR to reduce the influence of outliers.",
    when_to_use     = "When features have extreme outliers or long-tail distributions.",
    best_for        = "Features like Fare that bully others due to scale."
)

Scaler_LogMinMax = Scaler(
    method          = LogMinMaxMethod(),
    name            = "Log-MinMax",
    desc            = "Applies log transform followed by Min-Max scaling. Handles skew and outliers.",
    when_to_use     = "When features have extreme outliers or long-tail distributions.",
    best_for        = "Features like Fare that bully others due to scale, even after normalization."
)

