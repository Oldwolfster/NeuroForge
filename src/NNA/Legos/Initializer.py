import random
import math

class Initializer: #TODO rename StrategyInitializer
    """Encapsulates weight initialization strategies with proper bias handling."""

    def __init__(self, method, bias_method=None, name="Custom", desc="", when_to_use="", best_for=""):
        self.method = method  # Function for weight initialization
        self.bias_method = bias_method if bias_method else lambda: random.uniform(-1, 1)  # Default uniform bias
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for  # Best activation functions

    def __call__(self, num_inputs):
        """Generates initialized weights & bias given a shape."""
        weights = self.method(num_inputs)  # Generate weights
        bias = self.bias_method()     # Generate bias using the selected method
        return  [bias] + weights

    def __repr__(self):
        """Custom representation for debugging."""
        return self.name

def _dims(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)

def _generate(shape, generator):
    dims = _dims(shape)
    if len(dims) == 1:
        return [generator() for _ in range(dims[0])]
    elif len(dims) == 2:
        rows, cols = dims
        return [[generator() for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError("Only 1D and 2D shapes supported")

def rand_uniform(low, high, shape):
    return _generate(shape, lambda: random.uniform(low, high))

def rand_normal(mean, std, shape):
    return _generate(shape, lambda: random.gauss(mean, std))

def rand_scalar_uniform(low=-1, high=1):
    return random.uniform(low, high)

def rand_scalar_normal(mean=0, std=1):
    return random.gauss(mean, std)

# 🔹 **1. Uniform Random Initialization (Default)**
Initializer_Uniform = Initializer(
    method=lambda shape: rand_uniform(-1, 1, shape),
    bias_method=lambda: rand_scalar_uniform(-1, 1),  # Bias follows uniform distribution
    name="Uniform Random",
    desc="Assigns weights randomly from a uniform distribution [-1,1].",
    when_to_use="Useful for quick experimentation but may not be optimal for deep networks.",
    best_for="General use, but suboptimal for deep layers."
)

# 🔹 **2. Normal Distribution Initialization**
Initializer_Normal = Initializer(
    method=lambda shape: rand_normal(0, 1, shape),
    bias_method=lambda: rand_scalar_normal(0, 1),  # Bias follows normal distribution
    name="Normal Random",
    desc="Assigns weights using a normal distribution (mean=0, std=1).",
    when_to_use="Works well in simple networks but can lead to exploding/vanishing gradients in deep models.",
    best_for="General use."
)

# 🔹 **3. Xavier (Glorot) Initialization** - Optimized for **sigmoid/tanh**
Initializer_Xavier = Initializer(
    method=lambda shape: rand_uniform(
        -math.sqrt(6 / sum(_dims(shape))),
        math.sqrt(6 / sum(_dims(shape))),
        shape
    ),
    bias_method=lambda: 0,
    name="Xavier (Glorot)",
    desc="Scales weights based on number of inputs/outputs to maintain signal propagation.",
    when_to_use="Good for Sigmoid/Tanh activations in shallow networks.",
    best_for="Sigmoid, Tanh."
)

# 🔹 **4. He (Kaiming) Initialization** - Optimized for **ReLU-based activations**
Initializer_He = Initializer(
    method=lambda shape: rand_normal(0, math.sqrt(2 / _dims(shape)[0]), shape),
    bias_method=lambda: 0,                   # rand_scalar_normal(0, math.sqrt(2 / 2)),  # Bias follows weight scaling
    name="He (Kaiming)",
    desc="Optimized for ReLU, helps mitigate dying neurons.",
    when_to_use="Best for deep networks using ReLU to prevent vanishing gradients.",
    best_for="ReLU, Leaky ReLU."
)

# 🔹 **5. Small Random Values (Near Zero)**
Initializer_Tiny = Initializer(
    method=lambda shape: rand_normal(0, 0.01, shape),
    bias_method=lambda: rand_scalar_normal(0, 0.01),  # Bias also small
    name="Small Random",
    desc="Weights are initialized close to zero.",
    when_to_use="Used in some gradient-free methods or fine-tuning models.",
    best_for="Any activation."
)

# 🔹 **6. LeCun Initialization** - Optimized for **SELU activation**
Initializer_LeCun = Initializer(
    method=lambda shape: rand_normal(0, math.sqrt(1 / _dims(shape)[0]), shape),
    bias_method=lambda: 0,
    name="LeCun",
    desc="Similar to Xavier but optimized for self-normalizing networks.",
    when_to_use="Best when using SELU activation.",
    best_for="SELU, Tanh."
)

# 🔹 **7. Like my johnson **
Initializer_Huge = Initializer(
    method=lambda shape: _generate(shape, lambda: random.gauss(0, 1) * 10000.01 + 1111),
    bias_method=lambda: random.gauss(0, 0.01),  # Bias also small
    name="Large Random",
    desc="Weights are probably big.",
    when_to_use="Testing.",
    best_for="Any activation."
)

def xavier_kill_relu(shape):
    return rand_uniform(-2, -1, shape)

Initializer_KillRelu = Initializer(
    method=xavier_kill_relu,
    bias_method=lambda: -1.0,
    name="Xavier Kill ReLU",
    desc="Creates high chance of dead ReLU via negative initialization",
    when_to_use="Testing dead ReLU scenarios",
    best_for="Diagnostics"
)

# Determinist strategies below

# 🔹 **Strategic: Opposing Pairs** - Guarantees bidirectional exploration
def _opposing_pairs_init(shape):
    """
    For N weights, create N/2 random directions and their opposites.
    Guarantees ±coverage without RNG clustering.
    """
    dims = _dims(shape)

    # Handle 2D shapes (weights matrix)
    if len(dims) == 2:
        rows, cols = dims
        return [[_opposing_pairs_1d(cols) for _ in range(rows)]]

    # Handle 1D shapes (single layer weights)
    n_weights = dims[0]
    return _opposing_pairs_1d(n_weights)


def _opposing_pairs_1d(n_weights):
    """Generate opposing pairs for a 1D weight array"""
    n_pairs = n_weights // 2
    he_std = math.sqrt(2 / n_weights)

    # Generate half the weights
    half_weights = [random.gauss(0, he_std) for _ in range(n_pairs)]

    # Create opposing pairs
    opposing = [-w for w in half_weights]

    # Combine them
    weights = half_weights + opposing

    # Handle odd number of weights
    if n_weights % 2 == 1:
        weights.append(random.gauss(0, he_std))

    return weights


Initializer_OpposingPairs = Initializer(
    method=_opposing_pairs_init,
    bias_method=lambda: 0,
    name="Opposing Pairs",
    desc="Guarantees bidirectional exploration - creates N/2 random directions plus their opposites.",
    when_to_use="When you want guaranteed coverage of positive/negative weight space without RNG clustering.",
    best_for="ReLU, Leaky ReLU - prevents directional bias from random init."
)


# 🔹 **Strategic: Stratified Sampling** - Divides space into regions
def _stratified_init(shape):
    """
    Divides weight range into N equal regions, samples one point per region.
    Guarantees even distribution across the entire range.
    """
    n_weights = shape[0] if isinstance(shape, tuple) else shape
    he_std = math.sqrt(2 / n_weights)

    # Divide [-3*std, +3*std] into N equal regions (covers 99.7% of normal distribution)
    range_low = -3 * he_std
    range_high = 3 * he_std
    region_width = (range_high - range_low) / n_weights

    # Sample one point from each region
    weights = []
    for i in range(n_weights):
        region_start = range_low + i * region_width
        region_end = region_start + region_width
        # Random point within this region
        weight = random.uniform(region_start, region_end)
        weights.append(weight)

    # Shuffle to avoid positional bias
    random.shuffle(weights)
    return weights


Initializer_Stratified = Initializer(
    method=_stratified_init,
    bias_method=lambda: 0,
    name="Stratified Coverage",
    desc="Divides weight space into equal regions, samples once per region for guaranteed coverage.",
    when_to_use="When you want deterministic full-range coverage without random clustering.",
    best_for="ReLU, Leaky ReLU - reduces initialization variance across runs."
)


# 🔹 **Strategic: Orthogonal Basis** - Maximum separation between neurons
def _orthogonal_init(shape):
    """
    For small numbers of weights, place them at maximally separated angles.
    Like vertices of a regular polygon in high-dimensional space.
    """
    n_weights = shape[0] if isinstance(shape, tuple) else shape
    he_std = math.sqrt(2 / n_weights)

    if n_weights <= 1:
        return rand_normal(0, he_std, shape)

    # Divide circle into equal angles
    angles = [
        2 * math.pi * i / n_weights
        for i in range(n_weights)
    ]

    # Project onto weights using angle
    # Use He magnitude but orthogonal directions
    weights = [
        he_std * math.cos(angle)
        for angle in angles
    ]

    return weights


Initializer_Orthogonal = Initializer(
    method=_orthogonal_init,
    bias_method=lambda: 0,
    name="Orthogonal Basis",
    desc="Places neurons at maximally separated directions (equal angles) with He magnitude.",
    when_to_use="For small networks where you want maximum differentiation between neurons from start.",
    best_for="ReLU, Leaky ReLU - works best with 2-8 neurons per layer."
)
