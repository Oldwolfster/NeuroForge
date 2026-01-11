from src.NNA.legos.Activation import *
from src.NNA.legos.Loss import *
from src.NNA.legos.Optimizer import *
from src.NNA.utils.RamDB import RamDB
from pathlib import Path

class HyperParameters():
    def __init__(self):
        ############################################################
        # BATTLE Parameters are set here                           #
        ############################################################

        self.epochs_to_run           : int   = 20        # Number of times training run will cycle through all training data
        self.training_set_size       : int   = 12        # Qty of training data
        self.random_seed             : int   = 181467    #181467 #580636    # for seed 580636 - ONE EPOCH    #for seed 181026  DF LR 05 =9 but DF LR 4 = just 2 epochs    #for seed 946824, 366706 we got it in one!
        self.seed_replicates         : int   = 3         # Number of times to run each config with different random seeds (1 = no replication)
        self.nf_count                : int   = 2        # How many to display in NeuroForge
        self.display_train_data      : bool  = True      # Display the training data at the end of the rn.
        self.resume_batch            : bool  = False     # False = new batch, True = resume latest, or int = resume specific batch_id
        self.batch_name              : str   = ''
        self.batch_notes             : str   = ''
        #use_match_ui            : bool  = False  # Default to current workflow


        ############################################################
        # Optimizer-Specific Hyperparameters                       #
        ############################################################
        # Adam / AdamW / NAdam / AdaMax
        self.adam_beta1             : float = 0.9           # Momentum decay rate
        self.adam_beta2             : float = 0.999         # RMSprop decay rate
        self.adam_epsilon           : float = 1e-8          # Numerical stability

        # RMSprop / Adadelta
        self.rmsprop_beta           : float = 0.9           # Decay rate for moving average

        # Simplex
        self.simplex_growth_rate    : float = 1.05          # LR multiplier when stable
        self.simplex_decay_rate     : float = 0.5           # LR multiplier on explosion
        self.simplex_threshold      : float = None          # None = auto (input_max * 5)

        # Momentum
        self.momentum_beta          : float = 0.9           # Momentum coefficient



        self.dimensions = {
             "loss_function": [Loss_MSE, Loss_BCE]#, Loss_Huber, Loss_Hinge, Loss_LogCosh, Loss_HalfWit],
            # "hidden_activation": [Activation_Tanh, Activation_Sigmoid]#, Activation_LeakyReLU, Activation_ReLU,                                  Activation_NoDamnFunction],
            # "output_activation" : [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU, Activation_NoDamnFunction]
            # "initializer": "*",
            #"architecture": [[4, 4,3,4,4, 1], [2, 2, 1]],
            #"architecture": [[4,4,1],],                             #[1]],
            #"output_activation": [Activation_NoDamnFunction]
            #"seed":[1,2,3],
            #"loss":  Loss_HalfWit,
            #"optimizer": [Optimizer_SGD, Optimizer_Adam],
            # "batch_size": [1, 2, 4, 8, 999]
        }

        self.dimensions = {
             "loss_function": [Loss_MSE, Loss_BCE, Loss_Huber, Loss_Hinge, Loss_LogCosh, Loss_HalfWit],
             "hidden_activation": [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU,                                  Activation_NoDamnFunction],
             "output_activation" : [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU, Activation_NoDamnFunction],
             "initializer": "*",
            "architecture": [[4, 4,3,4,4, 1], [2, 2, 1]],
            "output_activation": [Activation_NoDamnFunction],
            "seed":[1,2,3],
            "loss":  Loss_HalfWit,
            "optimizer": [Optimizer_SGD, Optimizer_Adam],
             "batch_size": [1, 2, 4, 8, 999]
        }

        self.dimensions = {    "architecture": [[2, 5, 1], [2, 2, 1]], "optimizer": [ Optimizer_Nadam]}#Optimizer_SGD


        ############################################################
        # ARCHITECTURE WILDCARD CONFIGURATION                      #
        ############################################################
        # When dimensions = {"architecture": "*"}, these settings
        # control the comprehensive architecture search space.

        #ARCH_WILDCARD_MAX_LAYERS = 5    # Maximum hidden layers (0-5 = up to 5 hidden layers)
        #ARCH_WILDCARD_MAX_NEURONS = 16  # Maximum neurons per layer
        #ARCH_WILDCARD_MIN_NEURONS = 1   # Minimum neurons per layer

        self.db_ram: RamDB = RamDB()
        self.db_dsk: RamDB = RamDB(Path(__file__).parent.parent / "history" / "NF_history.db")


        #self.arenas = ['RepaymentFromCreditScore']
        self.arenas = ['CarValueFromMiles']
        #self.arenas = ['Titanic8']
        self.gladiators=['AutoForge','AutoForgeDup'] #,'TitanicOpus']
        self.gladiators = ['AutoForgeDup']  # ,'TitanicOpus']

    """
Cleanup list
MAYBE EARLY STOPPING
DONE RENAME ERROR SIGNAL TO ACCEPTED_BLAME
    1) Several popups need to be added
    3) DONE Thresholder visualization
    4) DONE Speed and jump to epoch controls
    5) Resolution-independent polish pass
    7) DONE Error history graph    
    9) DONE  color error analysis panel red/green
    10) DONE Model Banner    
    12) Double check in DisplayModel_NeuornScaler => 'output_surface = font.render(self.activation_function, True,' i
    13) Double check, do we need accumulated accepted blame in the Neuron table?
    15) DONEWe have max blame for both model and manager.... are we using both?  which should we use?
    8) DONE sample button
    11) DONE VCR tool tips
    6) DONE:LR sweep
    14) DONE: looks like it is drawing two output neurons - full architecture:  [-1, 3, 2, 2, -1] search yo -
    in displaymodel render() we should only be drawing visible neurons
GOAL OF6NF Refactor.  
    1) Resolution independence.
    2) DONE Remove pygamegui dependency        
    3) DONE rebuild arrows each frame... they were not following the neurons when a layer was  scrolled.
    3) DONE No RGB in code.
    4) DONE Limit of 2 models.
    
    6) show geometry for 2 input neurons.   decision boundries.
    7) DONE Ensure all button classes make sense - buttonmenu - buttonbase, probably more.    
    8) Menu working
    9) DONE - button_base - Pressed visual feedback (button depresses)
    
          
    GOAL OF REFACTOR NNA
    
1) DONE Treat bias like a weight in NNA
3)DONE  Binary Classification logic.
  - Detects: target has exactly 2 distinct values
  - Stores: label_low, label_high (unscaled)
  - Computes: threshold = (label_low + label_high) / 2
  - Classifies: unscaled_prediction > threshold → label_high, else label_low
  - Optionally warns: if activation/loss combo is suspect
4) Partial Recordings
5) DONE System for file paths
6) DONE DRY!  No recalcing in NF
7) Improve Early stopping
8) Clean serialization
8) DONE Fix name iteration -> sample everywhere.
9) DONE TRI, ModelInfo, Iteration have Dry issues - Place for everything and eveything in it-s place
10) Boycot the ridiculous underscore prefix pretending to be a scope modifier... give us option explicit and don't use it if you are stupid.

*** Hills i won't die on.  Punting for now - revise after clean refactor.
# Graceful handling of missing epoch-sample frames
# reorder sample
# training/test data
# Using LAG instead of storing every weight before and after.
# button_base - Hover visual feedback (color change)
#  button_base - Disabled state
# Would it pay to stop rebuilding fonts everywhere?
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