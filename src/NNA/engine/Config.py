from src.NNA.legos.Weight_initializer import *
from src.NNA.legos.Loss import *
from src.NNA.legos.Optimizer import *
from src.NNA.legos.Scaler import *
from src.NNA.engine._LegoAutoML import LegoAutoML
from src.NNA.legos.Activation import *


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
        from src.NNA.engine.TrainingRunInfo import RecordLevel
        self.update_from_batch_sweep(self.TRI.setup)


        #print(f"this data's problem type is '{self.TRI.training_data.problem_type}'")
        ok_to_print =  self.TRI.record_level != RecordLevel.NONE
        LegoAutoML(ok_to_print).apply(self, self.get_rules())
        self.finish_setup()

    def get_serialized_config(self) -> dict:
        """Return all config attributes as serializable dict."""
        excluded = {'TRI', 'scaler'}
        all_attrs = [attr for attr in dir(self) if not attr.startswith('_')]
        print(f"All Config attributes: {all_attrs}")
        print(f"print  self.loss_function ->{ self.loss_function}")
        config_dict = {}
        for attr in dir(self):
            if attr.startswith('_') or attr in excluded:
                continue

            value = getattr(self, attr, None)

            # Skip methods, but NOT strategy legos (which have .name or .var_name)
            if callable(value) and not (hasattr(value, 'name') or hasattr(value, 'var_name')):
                continue

            # Serialize lego instances
            if hasattr(value, 'var_name'):
                config_dict[attr] = value.var_name
            elif hasattr(value, 'name'):  # Strategy pattern legos
                config_dict[attr] = value.name
            elif isinstance(value, list):
                config_dict[attr] = str(value)
            else:
                config_dict[attr] = value  # Include None values too

        return config_dict


    def update_from_batch_sweep(self, setup):
        for key, value in setup.items():    # Loop through dimensions dictionary.
            if value is not None and hasattr(self, key): setattr(self, key, value)  # only update attribute if it exists

    def update_from_batch_sweep(self, setup):
        for key, value in setup.items():
            if value is not None and hasattr(self, key):
                # Deserialize string back to actual Lego instance
                deserialized_value = self.deserialize_lego(value)
                setattr(self, key, deserialized_value)

    def deserialize_lego(self, value):
        """Convert serialized Lego string back to instance."""
        if not isinstance(value, str):
            return value  # Already deserialized

        # Try to find matching Lego instance in imported modules
        try:
            return globals()[value]  # Look up by name in global scope
        except KeyError:
            return value  # Return as-is if not found


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
            (0, 602, {"input_scalers"       : Scaler_ZScore}                , "hidden_activation.name == 'Tanh'"),

            #Below are default settings if an above rule has not set an option
            (0, 6691, {"optimizer"          : Optimizer_SGD}                , "1 == 1"),
            (0, 6693, {"batch_size"         : 1}                            , "1 == 1"),
            (0, 6694, {"architecture"       : [8, 4, 1]}                    , "1 == 1"),
            (0, 6695, {"loss_function"      : Loss_MAE}                     , "1 == 1"),
            (0, 6696, {"hidden_activation"  : Activation_LeakyReLU}         , "1 == 1"),
            (0, 6697, {"weight_initializer" : Initializer_Xavier}           , "1 == 1"),
            (0, 6698, {"output_activation"  : Activation_NoDamnFunction}    , "1 == 1"),
            (0, 6699, {"target_scaler"      : Scaler_MinMax}                , "1 == 1"),
            (0, 6700, {"input_scalers"      : Scaler_Robust}                , "1 == 1"),
        ]
