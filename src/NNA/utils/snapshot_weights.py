from src.NNA.engine.Neuron import Neuron


def restore_weights(harvested: dict):
    """Inverse of harvest_weights -- crowns the network with a harvested weight set."""
    for layer in Neuron.layers:
        for neuron in layer:
            neuron.weights = harvested[neuron.nid].copy()

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

def harvest_weights() -> dict:
    """
    Returns {nid: weights.copy()} — plain data that survives neuron re-initialization.
    Bias rides along free (weights[0]). Called at the moment a new lowest MAE lands,
    so it captures the end-of-epoch weights of the record-setting epoch.
    """
    harvested = {}
    for layer in Neuron.layers:
        for neuron in layer:
            harvested[neuron.nid] = neuron.weights.copy()
    return harvested