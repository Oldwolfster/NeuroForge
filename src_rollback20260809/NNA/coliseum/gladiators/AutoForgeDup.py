import math
from typing import Tuple

from src.NNA.legos.Activation import *
from src.NNA.engine.BaseGladiator import Gladiator
from src.NNA.legos.Weight_initializer import *
from src.NNA.legos.Loss import *
from src.NNA.legos.Scaler import *
from src.NNA.legos.Optimizer import *
from src.NNA.engine.Config import Config
from src.NNA.engine.Neuron import Neuron


class AutoForge_TEMPLATE(Gladiator):
    """ AutoForge -  A ⚡imple Yet Powerful Neural Network ⚡
        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives
        🛡️ If you are having problems, comment everything out and try the 'smart defaults'
        """

    def configure_model(self, config: Config):
        """
        Optimized for the new 10-5-1 Architecture and 'Fat Tail' features.
        """

        #config.architecture = [2,1, 1]
        config.optimizer = Optimizer_SGD  # The "Turbo Start" version

        # We let the LR Sweep handle the aggressive start of NoHat
        #config.lr_specified = False

        # Kaiming (He) is mathematically the right partner for Leaky/ReLU
        config.weight_initializer = Initializer_He

        # LogCosh is great for regression, but if we are doing Titanic (Binary),
        # we need to be careful. I'll stick with your LogCosh play for the 'Smoothness' test.
        #config.loss_function = Loss_LogCosh2

        #config.batch_size = 7

        # MIX AND MATCH SCALERS:
        # Applying the "Math + Domain" logic we discussed.
        config.input_scalers = [
            Scaler_MinMax,  # Pclass (1, 2, 3) - Linear is fine
            Scaler_NONE,  # Sex (0, 1) - Already bounded
            Scaler_Robust,  # Age - Handles the 'old' outliers without crushing the 'young'
            Scaler_MinMax,  # SibSp - Small range (0-8), linear is fine
            Scaler_MinMax,  # Parch - Small range (0-6), linear is fine
            Scaler_LogMinMax,  # Fare - THE LOG PLAY. Squashes the luxury fares into the signal.
            Scaler_NONE,  # Embarked_S - One-hot
            Scaler_NONE,  # Embarked_C - One-hot
            Scaler_NONE  # Embarked_Q - One-hot
        ]

        # Target Scaler: If using LogCosh or MSE, MinMax on the target
        # helps keep the 'Blame' (gradient) within a predictable range.
        config.target_scaler = Scaler_MinMax
    """
    Between the above and the below, the following occurs:
        1) Config smart-defaults are set for anything not specified.
        2) Neurons Initialized and initial values set
        * NOTE: Data scaling will not yet have occurred when below runs.
    """
    def customize_neurons(self, config: Config):
        """ 🚀 Anything after initializing neurons
            🐉 but before training goes here  i.e manually setting a weight  """


    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  RECOMMENDED FUNCTIONS TO CUSTOMIZE  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Remove not_running__ prefix to activate  🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Not running be default  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹


    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Idiot proof features  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  THE KEY 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹

"""
1) Self setting LR
2) No exploding gradient
3) Does not allow incompatible output activtation function with loss functions
4) In fact, by default sets correct activation function for the loss function. 

☠️ 
👨‍🏫🍗🔥👑
🖼️  framed 
🔬  Microscope
🥂   toasting
🐉   dragon
💪
🚀💯🐶👨‍🍳
🐍💥❤️
😈   devil
😂   laugh
⚙️   cog
🔍
🧠   brain
🥩   steak
"""