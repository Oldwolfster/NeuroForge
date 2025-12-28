from src.NNA.Legos.Loss import StrategyLossFunction
from src.NNA.engine.RamDB import RamDB
from src.NNA.engine.TrainingData import TrainingData


class Config:
    """ Single(final) source of truth for the model configuration.
        The Gladiator's values will be stored here.
        In the even of batching, the value from dimensions will override a value in gladiator.
        Either way, this class is what the process uses as the configuration.
        Anything not set by the gladiator or dimensions will be set here
    """

    def __init__(self, TRI):
        self.TRI                                    = TRI   #Training Run info

        # Model Definition
        self.optimizer                              = None #: Optimizer
        self.learning_rate: float                   = None       # Read in beginning to instantiate  neurons with correct LR
        self.batch_size: int                        = None
        self.architecture: list                     = None
        self.weight_initializer: type               = None
        self.loss_function: StrategyLossFunction    = None
        self.hidden_activation: type                = None
        self.output_activation: type                = None
        self.target_scaler                          = None
        self.input_scalers                          = None

