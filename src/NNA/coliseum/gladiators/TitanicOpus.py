from src.NNA.Legos.ActivationFunctions import *
from src.NNA.engine.BaseGladiator import Gladiator
from src.NNA.Legos.WeightInitializers import *
from src.NNA.Legos.LossFunctions import *
from src.NNA.Legos.Scalers import *
from src.NNA.Legos.Optimizer import *
from src.NNA.engine.Config import Config
from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.convergence.ConvergenceDetector import ROI_Mode


class Titanic_V7(Gladiator):
    """
    Titanic V7: Full feature engineering for maximum accuracy.

    31 Inputs:
    - Core: Pclass, Sex, Age, FarePP, FamilySize, IsAlone, HasCabin, WomanOrChild, TicketFreq
    - Title (effect coded): 4 features
    - Embarked (effect coded): 2 features
    - Deck (effect coded): 7 features
    - AgeBin (effect coded): 4 features
    - Pclass_Sex (effect coded): 5 features
    """

    def configure_model(self, config: Config):
        config.architecture = [16, 8, 1]
        config.optimizer = Optimizer_Adam
        #config.learning_rate = 0.001
        config.weight_initializer = Initializer_He
        config.hidden_activation = Activation_LeakyReLU
        config.output_activation = Activation_Sigmoid
        config.loss_function = Loss_BCE
        config.batch_size = 32
        config.roi_mode = ROI_Mode.MOST_ACCURATE

        config.input_scalers = [
            # Core numeric features (9)
            Scaler_MinMax,  # Pclass (1-3)
            Scaler_NONE,  # Sex (already 0/1)
            Scaler_MinMax,  # Age (continuous)
            Scaler_LogMinMax,  # FarePP (skewed distribution)
            Scaler_MinMax,  # FamilySize (1-11)
            Scaler_NONE,  # IsAlone (binary)
            Scaler_NONE,  # HasCabin (binary)
            Scaler_NONE,  # WomanOrChild (binary)
            Scaler_MinMax,  # TicketFreq (1-7ish)

            # Title effect coding (4) - already -1/0/1
            Scaler_NONE,  # Title_F1
            Scaler_NONE,  # Title_F2
            Scaler_NONE,  # Title_F3
            Scaler_NONE,  # Title_F4

            # Embarked effect coding (2)
            Scaler_NONE,  # Embarked_F1
            Scaler_NONE,  # Embarked_F2

            # Deck effect coding (7)
            Scaler_NONE,  # Deck_F1
            Scaler_NONE,  # Deck_F2
            Scaler_NONE,  # Deck_F3
            Scaler_NONE,  # Deck_F4
            Scaler_NONE,  # Deck_F5
            Scaler_NONE,  # Deck_F6
            Scaler_NONE,  # Deck_F7

            # AgeBin effect coding (4)
            Scaler_NONE,  # AgeBin_F1
            Scaler_NONE,  # AgeBin_F2
            Scaler_NONE,  # AgeBin_F3
            Scaler_NONE,  # AgeBin_F4

            # Pclass_Sex effect coding (5)
            Scaler_NONE,  # PclassSex_F1
            Scaler_NONE,  # PclassSex_F2
            Scaler_NONE,  # PclassSex_F3
            Scaler_NONE,  # PclassSex_F4
            Scaler_NONE,  # PclassSex_F5
        ]

    def customize_neurons(self, config: Config):
        pass