class Neuron:
    """
    Represents a single neuron with weights, bias, an activation function, learning rates for each cog
    """
    layers = []             # Shared across all Gladiators, needs resetting per run
    neurons = []            # Shared across all Gladiators, needs resetting per run NOT INTENDED TO  BE PROTECTED
    output_neuron = None    # Shared access directly to the output neuron.

    def __init__(self, nid: int, num_of_weights: int, learning_rate: float, weight_initializer, layer_id: int = 0, activation = None):

        self.nid                = nid