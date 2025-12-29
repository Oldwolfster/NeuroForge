from src.NNA.engine.Neuron import Neuron


def snapshot_weights(from_suffix: str, to_suffix: str):
    """
    Copies weights and biases from one named attribute to another for all neurons.
    """
    for layer in Neuron.layers:
        for neuron in layer:
            from_weights = getattr(neuron, f"weights{from_suffix}")
            #from_bias = getattr(neuron, f"bias{from_suffix}")

            setattr(neuron, f"weights{to_suffix}", from_weights.copy())
            #setattr(neuron, f"bias{to_suffix}", from_bias)