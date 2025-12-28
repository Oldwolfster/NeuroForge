from src.NNA.Legos.Activation import *
from src.NNA.Legos.Initializer import *
from src.NNA.Legos.Loss import *
from src.NNA.Legos.Optimizer import *
from src.NNA.Legos.Scaler import *
from src.NNA.Legos._LegoAutoML import LegoAutoML


class Config:
    """ Single(final) source of truth for the model configuration.
        The Gladiator's values will be stored here.
        In the even of batching, the value from dimensions will override a value in gladiator.
        Either way, this class is what the process uses as the configuration.
        Anything not set by the gladiator or dimensions will be set here
    """

    def __init__(self, TRI):
        self.TRI                                            = TRI   #Training Run info

        # Model Definition
        self.learning_rate          : float                 = None       # Read in beginning to instantiate  neurons with correct LR
        self.batch_size             : int                   = None
        self.architecture           : list                  = None
        self.optimizer              :StrategyOptimizer      = None
        self.weight_initializer     : Initializer           = None
        self.loss_function          : StrategyLossFunction  = None
        self.hidden_activation      : StrategyActivation    = None
        self.output_activation      : StrategyActivation    = None
        self.target_scaler          : Scaler                = None
        self.input_scalers          : Scaler                = None
        self.scaler                 : MultiScaler           = MultiScaler(TRI.training_data)

    def autoML(self):
        self.update_from_batch_sweep(self.TRI.setup)

        print(self.TRI.training_data.raw_data)
        print(f"this data's problem type is '{self.TRI.training_data.problem_type}'")
        LegoAutoML().apply(self, self.get_rules(), self.TRI)
        self.finish_setup()

    def update_from_batch_sweep(self, setup):
        for key, value in setup.items():    # Loop through dimensions dictionary.
            if value is not None and hasattr(self, key): setattr(self, key, value)  # only update attribute if it exists

    def finish_setup(self):
        #TODO WHY IS THIS NOT WORKING  AND WHY DID WE NEED IT?  I KNOW WE DID... self.optimizer.config = self
        if self.input_scalers is not None:
            if isinstance(self.input_scalers, list):
                for i in range(self.TRI.training_data.input_count):
                    if i < len(self.input_scalers): self.scaler.set_input_scaler(self.input_scalers[i], i)
                    else:                           self.scaler.set_input_scaler(Scaler_NONE, i)
            else:                                   self.scaler.set_all_input_scalers(self.input_scalers)    # Single scaler for all inputs
        if self.target_scaler:                      self.scaler.set_target_scaler(self.target_scaler)
        if self.architecture[-1] != 1:              self.architecture.append(1) #Ensure one output neuron

    def get_rules(self):
        #   Allow overwrite, priority, field to set, value, condition to set it.
        return [
            # First choose loss (based on problem type or custom override)
            #TODO match these to the new class names
            (0, 200, {"loss_function"       : Loss_BCE}                     , "TRI.training_data.is_binary_decision"),
            (0, 201, {"loss_function"       : Loss_MSE}                     , "not TRI.training_data.is_binary_decision"),
            (0, 300, {"output_activation"   : Activation_Sigmoid}           , "loss_function.name == 'Binary Cross-Entropy'"),
            (0, 301, {"output_activation"   : Activation_NoDamnFunction}    , "loss_function.name == 'Mean Squared Error'"),
            (0, 302, {"output_activation"   : Activation_NoDamnFunction}    , "loss_function.name == 'Hinge Loss'"),
            (1, 500, {"target_scaler"       : Scaler_MinMax_Neg1to1}        , "output_activation.name == 'Tanh'"),
            (1, 501, {"target_scaler"       : Scaler_MinMax}                , "output_activation.name == 'Sigmoid'"),
            (1, 502, {"target_scaler"       : Scaler_MinMax_Neg1to1}        , "loss_function.name == 'Hinge Loss'"),
            (0, 600, {"weight_initializer"  : Initializer_He}               , "hidden_activation.name == 'LeakyReLU'"),
            (0, 601, {"weight_initializer"  : Initializer_He}               , "hidden_activation.name == 'ReLU'"),

            #Below are default settings if an above rule has not set an option
            (0, 6691, {"optimizer"          : Optimizer_SGD}                , "1 == 1"),
            (0, 6693, {"batch_size"         : 1}                            , "1 == 1"),
            (0, 6694, {"architecture"       : [2, 1]}                       , "1 == 1"),
            (0, 6695, {"loss_function"      : Loss_MAE}                     , "1 == 1"),
            (0, 6696, {"hidden_activation"  : Activation_LeakyReLU}         , "1 == 1"),
            (0, 6697, {"weight_initializer" : Initializer_Xavier}           , "1 == 1"),
            (0, 6698, {"output_activation"  : Activation_NoDamnFunction}    , "1 == 1"),
            (0, 6699, {"target_scaler"      : Scaler_MinMax}                , "1 == 1"),
            (0, 6700, {"input_scalers"      : Scaler_Robust}                , "1 == 1"),
        ]
