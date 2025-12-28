from src.NNA.engine.RamDB import RamDB
from src.NNA.Legos.Optimizer import *
from src.NNA.Legos.Loss import *
from src.NNA.Legos.Initializer import *
from pathlib import Path

class HyperParameters():
    def __init__(self):
        ############################################################
        # BATTLE Parameters are set here                           #
        ############################################################

        self.epochs_to_run           : int   = 30     # Number of times training run will cycle through all training data
        self.training_set_size       : int   = 2        # Qty of training data
        self.random_seed             : int   = 0    #181467 #580636    # for seed 580636 - ONE EPOCH    #for seed 181026  DF LR 05 =9 but DF LR 4 = just 2 epochs    #for seed 946824, 366706 we got it in one!
        self.seed_replicates         : int   = 3         # Number of times to run each config with different random seeds (1 = no replication)
        self.nf_count                : int   = 2        # How many to display in NeuroForge
        self.display_train_data      : bool  = True      # Display the training data at the end of the rn.
        self.resume_batch            : bool  = False     # False = new batch, True = resume latest, or int = resume specific batch_id
        self.batch_name              : str   = ''
        self.batch_notes             : str   = ''
        #use_match_ui            : bool  = False  # Default to current workflow

        self.dimensions = {
            # "loss_function": [Loss_MSE, Loss_BCE, Loss_Huber, Loss_Hinge, Loss_LogCosh, Loss_HalfWit],
            # "hidden_activation": [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU,                                  Activation_NoDamnFunction],
            # "output_activation" : [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU, Activation_NoDamnFunction]
            # "output_activation" : [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU, Activation_NoDamnFunction]
            # "initializer": "*",
            # "architecture": [[4, 4, 1], [2, 2, 1]],
            #"seed":[1,2,3],
            "loss":  Loss_HalfWit,
            "optimizer": [Optimizer_SGD, Optimizer_Adam],
            # "batch_size": [1, 2, 4, 8, 999]
        }




        ############################################################
        # ARCHITECTURE WILDCARD CONFIGURATION                      #
        ############################################################
        # When dimensions = {"architecture": "*"}, these settings
        # control the comprehensive architecture search space.

        #ARCH_WILDCARD_MAX_LAYERS = 5    # Maximum hidden layers (0-5 = up to 5 hidden layers)
        #ARCH_WILDCARD_MAX_NEURONS = 16  # Maximum neurons per layer
        #ARCH_WILDCARD_MIN_NEURONS = 1   # Minimum neurons per layer

        self.db_ram = RamDB()
        self.db_dsk = RamDB(Path(__file__).parent.parent / "history" / "NF_history.db")


        self.arenas = ['RepaymentFromCreditScore']
        self.gladiators=['AutoForge','TitanicOpus']



    """
    GOAL OF REFACTORS
1) Treat bias like a weight in NNA
2) Test set separate from training
3) Binary Classification logic.
  - Detects: target has exactly 2 distinct values
  - Stores: label_low, label_high (unscaled)
  - Computes: threshold = (label_low + label_high) / 2
  - Classifies: unscaled_prediction > threshold → label_high, else label_low
  - Optionally warns: if activation/loss combo is suspect
4) Partial Recordings
5) System for file paths
6) DRY!  No recalcing in NF
7) Early stopping
8) Clean serialization
8) Fix name iteration -> sample everywhere.
9) TRI, ModelInfo, Iteration have Dry issues - Place for everything and eveything in it-s place
- For DB bound anything.
- IterationRecord
- EpochRecord
- RunRecord
10) Leverge RamDB for all sql.(eliminate all or nearly all sql)
11) Boycot the ridiculous underscore prefix pretending to be a scope modifier... give us option explicit and don't use it if you are stupid.
"""

"""

#######################################################################
######################### Regression ##################################
#######################################################################
'Predict_Income_2_Inputs'
,'California_Housing'
,'California_HousingUSD'
,'One_Giant_Outlier'
,'One_Giant_OutlierExplainable'
,'Nested_Sine_Flip'
,'Chaotic_Function_Prediction'
,'Piecewise_Regime'
,'Adversarial_Noise'
,'MultiModal_Nonlinear_Interactions'
,'MultiModal_Temperature'

,'Delayed_Effect_BloodSugar'
,'Predict_EnergyOutput__From_Weather_Turbine'
,'Chaotic_Solar_Periodic'
,'Custom_Function_Recovery'
,'Deceptive_Multi_Regime_Entangler'
,'Predict_Income_2_Inputs_5Coefficents'
,'Predict_Income_2_Inputs_Nonlinear'
,'Predict_Income_Piecewise_Growth'
,'Customer_Churn_4X3'
,'AutoNormalize_Challenge'
,'Red_Herring_Features'
,'CarValueFromMiles'
,'Predict_MedicalCost_WithOutliers'
,'Target_Drift_Commodity'
,'Redundant_Features'

#######################################################################
######################### Binary Decision #############################
#######################################################################
,'Titanic'
,'SimpleBinaryDecision'
,'DefaultRisk__From_Income_Debt'
,'Bit_Flip_Memory'
,'Parity_Check'
,'XOR_Floats'
,'Sparse_Inputs'
,'Circle_In_Square'
,'XOR'
,'Moons'
,'Iris_Two_Class'

"""