from src.NNA.legos.Activation import *
from src.NNA.legos.Weight_initializer import *
from src.NNA.legos.Loss import *
from src.NNA.legos.Optimizer import *
from src.NNA.utils.db_conn import db_conn

from pathlib import Path


import math

class HyperParameters():
    @property
    def nf_db(self):
        if self.nf_db_handle is None: self.nf_db_handle = db_conn()
        return self.nf_db_handle

    # ArenaSettings.py  method: nf_db  NEW:  setter — build_hyper assigns the explicit path
    @nf_db.setter
    def nf_db(self, db):
        self.nf_db_handle = db

    def __init__(self):
        self.nf_db_handle           = None                                                  # NEW
        self.reporter               = None  #Shard report for communicating to UI process

        ############################################################
        # BATTLE Parameters are set here                           #
        ############################################################
        self.epochs_to_run           : int   = 50        # Number of times training run will cycle through all training data
        self.training_set_size       : int   = 120       # Qty of training data
        self.random_seed             : int   = 180589 #980963   # for seed 580636 - ONE EPOCH    #for seed 181026  DF LR 05 =9 but DF LR 4 = just 2 epochs    #for seed 946824, 366706 we got it in one!
        #self.gladiators = ['AutoForgeNoScaler']  # ,'TitanicOpus']
        self.gladiators = ['AutoForge']  # ,'TitanicOpus']
        self.arenas = ['RepaymentFromCreditScore']

        self.seed_replicates         : int   = 1         # Number of times to run each config with different random seeds (1 = no replication)
        self.nf_count                : int   = 2        # How many to display in NeuroForge
        self.record_level            : str   = None
        self.sweep_precision         : int   = 1
        self.display_train_data      : bool  = True      # Display the training data at the end of the rn.
        self.resume_batch            : bool  = False     # False = new batch, True = resume latest, or int = resume specific batch_id
        self.batch_name              : str   = ''
        self.batch_notes             : str   = ''
       #self.neuro_FORGE                     = [0]          #NORMAL
        self.neuro_FORGE                     = [0]    #To see AdaMax spank Nadam
        #use_match_ui            : bool  = False  # Default to current workflow


        ############################################################
        # Optimizer-Specific Hyperparameters                       #
        ############################################################
        # Adam / AdamW / NAdam / AdaMax RMSprop / Adadelta
        self.optimizer_beta1        : float = 0.9           # Momentum decay rate
        self.optimizer_beta2        : float = 0.999         # RMSprop decay rate
        self.optimizer_epsilon      : float = 1e-8          # Numerical stability
        self.early_stopping_thresh  : float = 0.001

        ############################################################
        # ARCHITECTURE WILDCARD CONFIGURATION                      #
        ############################################################
        # When dimensions = {"architecture": "*"}, these settings
        # control the comprehensive architecture search space.

        #ARCH_WILDCARD_MAX_LAYERS = 5    # Maximum hidden layers (0-5 = up to 5 hidden layers)
        #ARCH_WILDCARD_MAX_NEURONS = 16  # Maximum neurons per layer
        #ARCH_WILDCARD_MIN_NEURONS = 1   # Minimum neurons per layer

        #self.dimensions = {"architecture": [[4, 1]],#, [9, 6, 1]],#"architecture": [[4, 4,3,4,4, 1], [2, 2, 1]],
        self.dimensions = {"architecture": [[2, 1]],  # , [9, 6, 1]],#"architecture": [[4, 4,3,4,4, 1], [2, 2, 1]],
                           "batch_size": [50,1],
                           "seed":      [980963, 230936, 149314, 401318, 180589],
            "loss_function":[Loss_HalfWit,   Loss_BCE, Loss_SquaredHinge, Loss_MSE],#[Loss_SquaredHinge, Loss_Hinge],
            "optimizer":[Optimizer_Opus]#, Optimizer_RProp, Optimizer_Adam]#   , Optimizer_RAdam] # [ Optimizer_RAdam]#, Optimizer_MoonPhase]#####################,Optimizer_SGD]
                           }
        self.dimensions = {
            "architecture": [[2, 1]],


            #"seed": [649691],
                    #+ list(range(700001, 700046)),  # 5 known + 45 new = 50
            "learning_rate":[.01],
            "batch_size": [50, 1],
            "optimizer": [Optimizer_Fable],
        }
        self.dimensions = {}


        self.arenas = ['Predict_Income_2_Inputs_Nonlinear']
        self.arenas = ['Moons']
        self.arenas = ['Titanic8']
        self.arenas = ['SingleInput_CreditScore']
        self.arenas = ['Real_CaliforniaHousing']
        self.arenasAll=[
            "Adversarial_Noise",
            "Arena_CenteredData",
            "Arena_MixedMagnitudes",
            "Arena_NonlinearCompression",
            "Arena_OutlierSensitivity",
            "Arena_ZeroVariance",
            "AutoNormalize_Challenge",
            "Bit_Flip_Memory",
            "California_Housing",
            "California_HousingUSD",
            "CarValueFromMiles",
            "Chaotic_Function_Prediction",
            "Chaotic_Solar_Periodic",
            "Circle_In_Square",
            "Complex_Market_Sentiment_Swing",
            "CreditScoreRegression",
            "CreditScoreRegressionNeedsBias",
            "CustomerChurn",
            "Customer_Churn_4X3",
            "Custom_Function_Recovery",
            "Deceptive_Multi_Regime_Entangler",
            "DefaultRisk__From_Income_Debt",
            "Delayed_Effect_BloodSugar",
            "DiseaseRisk__From_HealthMetrics",
            "Hidden_Regime_Shifter",
            "Hidden_Switch_Power",
            "HouseValue_SqrFt",
            "Income__Experience_CompanyRevenue",
            "Iris_Two_Class",
            "Local_Minima",
            "Manual",
            "Moons",
            "MultiModal_Nonlinear_Interactions",
            "MultiModal_Temperature",
            "Nested_Sine_Flip",
            "One_Giant_Outlier",
            "One_Giant_OutlierExplainable",
            "Parity_Check",
            "Pathological_Discontinuous_Chaos",
            "Piecewise_Regime",
            "Predict_EnergyOutput__From_Weather_Turbine",
            "Predict_FuelCost__From_MilesDriven_GasPrice",
            "Predict_Income_1_Input",
            "Predict_Income_2_Inputs",
            "Predict_Income_2_InputsFeatureEngineer",
            "Predict_Income_2_Inputs_5Coefficents",
            "Predict_Income_2_Inputs_Multiplicative",
            "Predict_Income_2_Inputs_NoImpactFrom2nd",
            "Predict_Income_2_Inputs_Nonlinear",
            #"Predict_Income_3_Inputs",
            "Predict_Income_Piecewise_Growth",
            "Predict_MedicalCost_WithOutliers",
            "Predict_TrafficFlow__From_Weather_Time_Events",
            "Redundant_Features",
            "Red_Herring_Features",
            "Regime_Trigger_Switch",
            "RepaymentFromCreditScore",
            "Salary2Inputs",
            "Salary2InputsNonlinear",
            #"Salary2Inputs_B",
            #"Salary2Inputs_C",
            #"SalaryExperienceRegressionNeedsBias",
            "SingleInput_CreditScore",
            "Sparse_Inputs",
            "StockPrice__From_Indicators",
            "Target_Drift_Commodity",
            "Titanic",
            #"Titanic2",
            #"Titanic3",
            #"Titanic4",
            #"Titanic5",
            #"Titanic6",
            #"Titanic7",
            "Titanic8",
            "XOR",
        ]


    """


- Pytorch export (unless TF would be WAY easier)
- graph improvements - have tons built... just need to make available.
- data reporter - mini little crystal report taking a sql statement
- Early Stopping 
- Test all dimensions will batch
- DONE Clean up Arenas
- Rename batch table schema
- DONE Hook optimizer parameters to the hyperparameters 

- add the geometry
- Finish resume existing batch... existing code is not close

DEFECT LIST:
output neuron's pop up does not have blame sources
Seed replicates is not working
ENSURE optimizer popup does not change fields


Cleanup list DONE!!!!
    1) DONE Several popups need to be added
    2) DONE RENAME ERROR SIGNAL TO ACCEPTED_BLAME
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
G
# Graceful handling of missing epoch-sample frames
# reorder sample
# training/test data
# Using LAG instead of storing every weight before and after.
# button_base - Hover visual feedback (color change)
#  button_base - Disabled state
# Would it pay to stop rebuilding fonts everywhere?
"""

