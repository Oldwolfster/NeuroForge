# pylint: disable=no-self-use
from abc                       import ABC
from json                      import dumps
from src.NNA.Legos.Activation import *
from src.NNA.Legos.Optimizer import *
from src.NNA.engine.Config import Config
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
#from src.NNA.engine.VCRRecorder          import VCR
#from src.NNA.engine.Config       import Config
from src.NNA.engine.Neuron       import Neuron
from datetime                    import datetime

#from src.NNA.engine.Utils_DataClasses import Iteration
#from src.NNA.Legos.WeightInitializers import *



class Gladiator(ABC):
    """
    💥 NOTE: The gradient is inverted from the traditional way of thinking.
    Abstract base class for creating Gladiators (neural network models).
    Goal: Give child gladiator class as much power as possible, requiring as little responsibility
    as possible, while allowing for overwriting any step of the process.

    There are three main sections:
    1) Initialization - Preps everything for the framework and gladiator
    2) Training Default Methods - (Forward and Backwards Pass) available for free, but overwritable for experimentation.
    3)

    """

    def __init__(self,  TRI: TrainingRunInfo):
        self.TRI                    = TRI  # TrainingRunInfo
        self.config                 = TRI.config
        self.db                     = TRI.db
        self.training_data          = TRI.training_data         # Only needed for sqlMgr ==> self.ramDb = args[3]
        self.sample                 = 0
        self.epoch                  = 0
        self.blame_calculations     = []
        self.weight_calculations    = []
        self.finalize_setup()

    def finalize_setup(self):
        self.configure_model(self.config)                   # Typically overwritten in child (gladiator) class.
        self.config.autoML()







    def configure_model(self, config: Config):   pass  # Typically overwritten in child  class.

    def customize_neurons(self, config: Config): pass  # Typically overwritten in child  class.


