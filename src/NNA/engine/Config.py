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
        self.input_scalers          : dict                  = None
        self.scaler                 : MultiScaler           = MultiScaler(TRI.training_data.input_count)

    def autoML(self):
        from src.NNA.engine.TrainingRunInfo import RecordLevel
        self.update_from_batch_sweep(self.TRI.setup)
        self.pull_arena_scalers()                              # NEW  frozen map -> self.input_scalers (dict), before finish_setup
        ok_to_print =  self.TRI.record_level != RecordLevel.NONE
        LegoAutoML(ok_to_print).apply(self, self.get_rules())
        self.finish_setup()

    def pull_arena_scalers(self):
        batch_map = getattr(self.TRI.hyper, "input_scalers", None)
        #print(f"🎯 PULL: hyper.input_scalers = {batch_map}")           # TEMP
        if not isinstance(batch_map, dict) or not batch_map:            return
        arena = self.TRI.training_data.arena_name
        slice = batch_map.get(arena)
        #print(f"🎯 PULL: arena='{arena}'  slice={slice}")              # TEMP
        if slice:   self.input_scalers = slice             # a {field: name} dict — finish_setup applies it by name


    def get_serialized_config(self) -> dict:
        """Return all config attributes as serializable dict."""
        excluded = {'TRI', 'scaler'}
        all_attrs = [attr for attr in dir(self) if not attr.startswith('_')]
        print(f"All Config attributes: {all_attrs}")
        print(f"print  self.loss_function ->{ self.loss_function}")
        config_dict = {}
        for attr in dir(self):
            if attr.startswith('_') or attr in excluded:        continue
            value = getattr(self, attr, None)
            if callable(value) and not (hasattr(value, 'name') or hasattr(value, 'var_name')):                continue

            # Serialize lego instances
            if hasattr(value, 'var_name'):
                config_dict[attr] = value.var_name
            elif hasattr(value, 'name'):  # Strategy pattern legos
                config_dict[attr] = value.name
            elif isinstance(value, dict):  # NEW  resolved {field: scaler} map
                import json  # NEW
                config_dict[attr] = json.dumps(value)
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

    def deserialize_legoOLD(self, value):
        """Convert serialized Lego string back to instance."""
        if not isinstance(value, str):
            return value  # Already deserialized

        # Try to find matching Lego instance in imported modules
        try:
            return globals()[value]  # Look up by name in global scope
        except KeyError:
            return value  # Return as-is if not found

    def deserialize_lego(self, value):
        """Serialized name back to instance. A name that resolves to nothing is fatal."""
        if not isinstance(value, str):          return value    # already deserialized
        if value in globals():                  return globals()[value]
        if value in Scaler.by_name:             return Scaler.by_name[value]
        raise ValueError(f"HOUSTON: cannot deserialize lego '{value}'.\n"
                         f"Not a symbol in Config's globals and not in Scaler.by_name.\n"
                         f"A string here becomes a scaler with no .name and crashes downstream.")

    # ═══════════════════════════════════════════════════════════════════════
    # finish_setup SCALERS SCALERS SCALERS SCALERS SCALERS SCALERS
    # ═══════════════════════════════════════════════════════════════════════

    def finish_setupOLD(self):
        #TODO WHY IS THIS NOT WORKING  AND WHY DID WE NEED IT?  I KNOW WE DID... self.optimizer.config = self
        if self.input_scalers is not None:
            if   isinstance(self.input_scalers, dict): self.apply_scalers_by_name(self.input_scalers)
            elif isinstance(self.input_scalers, list):
                for i in range(self.TRI.training_data.input_count):
                    if i < len(self.input_scalers): self.scaler.set_input_scaler(self.input_scalers[i], i)
                    else:                           self.scaler.set_input_scaler(Scaler_NONE, i)
            else:                                   self.scaler.set_all_input_scalers(self.input_scalers)
        if self.target_scaler:                      self.scaler.set_target_scaler(self.target_scaler)
        if self.architecture[-1] != 1:              self.architecture.append(1)


    def finish_setup(self):
        #TODO WHY IS THIS NOT WORKING  AND WHY DID WE NEED IT?  I KNOW WE DID... self.optimizer.config = self
        if self.input_scalers:                      self.apply_scalers_by_name(self.input_scalers)
        if self.target_scaler:                      self.scaler.set_target_scaler(self.target_scaler)
        if self.architecture[-1] != 1:              self.architecture.append(1)

    # Config.py  method: apply_scalers_by_name  Update: read the self-registered map off Scaler, not a Config attribute
    def apply_scalers_by_name(self, name_map):
        """The resolver ships {field: scaler_name}. Match to columns by label, not position.
        Field order at launch and at run both come from the arena, so names can't drift."""
        labels = self.TRI.training_data.feature_labels or []
        for i in range(self.TRI.training_data.input_count):
            field  = labels[i] if i < len(labels) else None
            scaler = Scaler.by_name.get(name_map.get(field)) if field else None
            self.scaler.set_input_scaler(scaler or Scaler_NONE, i)


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
            #(0, 602, {"input_scalers"       : Scaler_ZScore}                , "hidden_activation.name == 'Tanh'"),

            #Below are default settings if an above rule has not set an option
            (0, 6691, {"optimizer"          : Optimizer_SGD}                , "1 == 1"),
            (0, 6693, {"batch_size"         : 1}                            , "1 == 1"),
            (0, 6694, {"architecture"       : [2, 1]}                    , "1 == 1"),
            (0, 6695, {"loss_function"      : Loss_MAE}                     , "1 == 1"),
            (0, 6696, {"hidden_activation"  : Activation_LeakyReLU}         , "1 == 1"),
            (0, 6697, {"weight_initializer" : Initializer_Xavier}           , "1 == 1"),
            (0, 6698, {"output_activation"  : Activation_NoDamnFunction}    , "1 == 1"),
            (0, 6699, {"target_scaler"      : Scaler_MinMax}                , "1 == 1"),
            #(0, 6700, {"input_scalers"      : Scaler_Robust}                , "1 == 1"),
        ]
