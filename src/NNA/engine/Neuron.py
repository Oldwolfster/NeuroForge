from src.NNA.Legos.Activation import Activation_NoDamnFunction, StrategyActivation


class Neuron:
    """  Represents a single neuron with weights, bias, an activation function, learning rates for each cog
         Weights: [0] is bias, [1:] are connection weights
    """
    layers          = []    # Shared across all Gladiators, needs resetting per run
    neurons         = []    # Shared across all Gladiators, needs resetting per run
    output_neuron   = None  # Shared access directly to the output neuron.

    def __init__(self, nid: int, num_inputs: int, learning_rate: float, weight_initializer, layer_id: int, activation: StrategyActivation):
        self.nid            = nid
        self.layer_id       = layer_id
        self.weights        = weight_initializer(num_inputs)  # Returns list of length num_inputs + 1
        self.weights_before = self.weights.copy()
        self.neuron_inputs  = [0.0] * len(self.weights) # Don't think i need this.  I was wrong

        # Per-weight state
        self.learning_rates = [learning_rate] * len(self.weights)
        #self.m = [0.0] * len(self.weights)  # Adam momentum
        #self.v = [0.0] * len(self.weights)  # Adam variance
        self.accumulated_accepted_blame = [0.0] * len(self.weights)
        #self.t = 0  # Timestep counter for optimizer

        # Activation
        self.activation             = activation
        self.raw_sum                = 0.0
        self.activation_value       = 0.0
        self.activation_gradient    = 0.0
        self.error_signal           = 0.0

        # Register in class collections
        Neuron.neurons.append(self)
        while len(Neuron.layers) <= layer_id:  Neuron.layers.append([])  # Ensure layers list is large enough to accommodate this layer_id
        Neuron.layers[layer_id].append(self)
        self.position = len(Neuron.layers[layer_id]) - 1
        if layer_id == len(Neuron.layers) - 1: Neuron.output_neuron = self

    @property
    def num_inputs(self):  return len(self.weights) - 1

    def activate(self):
        self.activation_value = self.activation(self.raw_sum)
        self.activation_gradient = self.activation.derivative(self.raw_sum)