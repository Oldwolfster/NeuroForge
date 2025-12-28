import random

def set_seed(seed) -> int:
    """ Sets random seed for numpy & Python's random module.
        If hyperparameters has seed value uses it for repeatabilty.
        IF not, generates randomly
    """
    if seed == 0:        seed = random.randint(1, 999999)
    # np.random.seed(seed)
    random.seed(seed)
    return seed
