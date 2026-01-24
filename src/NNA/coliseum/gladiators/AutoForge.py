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
        ✅ Flexible architecture with preconfigured alternatives
        🛡️ If you are having problems, comment everything out and try the 'smart defaults'
        """

    def configure_model(self, config: Config):
        """ 👉  Anything prior to initializing neurons goes here
            💪  For example setting config options.        """

        #config.architecture            = [2,4,1]
        #config.optimizer                = Optimizer_Adam_NoHat
        config.learning_rate           = 0.5
        #config.weight_initializer      = Initializer_Xavier
        config.hidden_activation       = Activation_Tanh
        #config.output_activation       = Activation_Sigmoid
        #config.loss_function           = Loss_BCE
        config.batch_size              = 4
        #config.roi_mode                = ROI_Mode.MOST_ACCURATE    #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE
        config.input_scalers           = Scaler_ZScore                                 # All inputs same scaler
        #config.input_scalers           = [Scaler_MinMax, Scaler_MinMax, Scaler_MinMax, Scaler_MinMax,Scaler_MinMax,Scaler_MinMax,Scaler_MinMax,Scaler_MinMax, Scaler_Robust]
        #config.target_scaler           = Scaler_NONE #Scaler_NONE # Scaler_MinMax
        """config.input_scalers = [
            Scaler_MinMax,  # Pclass
            Scaler_NONE,  # Sex already 0 and 1
            Scaler_MinMax,  # Age
            Scaler_MinMax,  # SibSp # prob robust
            Scaler_MinMax,  # Parch # Scaler_Robust
            Scaler_LogMinMax,  # Fare - will switch to Scaler_LogMinMax
            Scaler_NONE,  # Embarked_S - one hot encoded
            Scaler_NONE,  # Embarked_C - one hot encoded
            Scaler_NONE  # Embarked_Q - one hot encoded
        ]
         GBS Gospel
        config.input_scalers = [
            Scaler_MinMax,  # Pclass
            Scaler_NONE,  # Sex already 0 and 1
            Scaler_MinMax,  # Age
            Scaler_MinMax,  # SibSp # prob robust
            Scaler_MinMax,  # Parch # Scaler_Robust
            Scaler_LogMinMax,  # Fare - will switch to Scaler_LogMinMax
            Scaler_NONE,  # Embarked_S - one hot encoded
            Scaler_NONE,  # Embarked_C - one hot encoded
            Scaler_NONE  # Embarked_Q - one hot encoded
        ]
        """
                                         # Target scaling
    """
    Between the above and the below, the following occurs:
        1) Config smart-defaults are set for anything not specified.
        2) Neurons Initialized and initial values set
        * NOTE: Data scaling will not yet have occurred when below runs.
    """
    def customize_neurons(self, config: Config):
        """ 🚀 Anything after initializing neurons
            🐉 but before training goes here  i.e manually setting a weight  """
        #Neuron.output_neuron.set_activation(Activation_NoDamnFunction)  #How to change a neurons activation initialization occured
        #config.db.query_print("SELECT * FROM  Neuron LIMIT 1")

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