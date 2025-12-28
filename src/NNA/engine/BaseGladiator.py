# pylint: disable=no-self-use
from abc                       import ABC
from json                      import dumps
from src.NNA.Legos.ActivationFunctions import *
from src.NNA.Legos.Optimizers import *
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
from src.NNA.engine.VCRRecorder          import VCR
from src.NNA.engine.Config       import Config
from src.NNA.engine.Neuron       import Neuron
from datetime                    import datetime
from src.NNA.engine.Utils_DataClasses import ez_debug
from src.NNA.engine.Utils_DataClasses import Iteration
from src.NNA.Legos.WeightInitializers import *
from typing                      import List, Tuple


class Gladiator(ABC):
    """
    💥 NOTE: The gradient is inverted from the traditional way of thinking.
    Abstract base class for creating Gladiators (neural network models).
    Goal: Give child gladiator class as much power as possible, requiring as little responsibility
    as possible, while allowing for overwriting any step of the process.

    There are three main sections:
    1) Training Default Methods - available for free, but overwritable for experimentation.
    2) Training Framework - does the brute force tasks of the arena - not intended for overriding
    3) Initialization - Preps everything for the framework and gladiator
    """
    #def __init__(self,  config: Config):
    def __init__(self,  TRI: TrainingRunInfo):
        self.TRI                    = TRI  # TrainingRunInfo
        self.config                 = TRI.config
        self.db                     = TRI.db
        self.training_data          = TRI.training_data         # Only needed for sqlMgr ==> self.ramDb = args[3]
        self.VCR                    = VCR(TRI, Neuron.neurons)  # Args3, is ramdb
        self.total_iterations       = 1                         # Timestep for optimizers such as adam
        self.number_of_epochs       = TRI.max_epochs
        self.iteration              = 0
        self.epoch                  = 0
        self.too_high_adjst         = self.training_data.input_max * 5  # TODO make '5' a hyperparamter
        self.max_adj                = self.training_data.everything_max_magnitude()
        self.blame_calculations     = []
        self.weight_calculations    = []
        self.convergence_phase      = "watch"
        self.finalize_setup()

    ################################################################################################
    ################################ SECTION 1 - pipeline ##########################################
    ################################################################################################

    def finalize_setup(self):
        self.configure_model(self.config)                   # Typically overwritten in child  class.
        self.config.smartNetworkSetup(self.TRI)
        self.initialize_neurons(
            architecture        = self.config.architecture.copy(),   # Avoid mutation
            initializers        = [self.config.weight_initializer],         # <- List of 1 initializer
            hidden_activation   = self.config.hidden_activation,
            output_activation   = self.config.output_activation or self.config.loss_function.recommended_output_activation
        )
        # ez_debug(outputNNN=Neuron.output_neuron.activation.bd_defaults)
        self.customize_neurons  (self.config)                 # Typically overwritten in child  class.
        self.binary_decision_setup()

    def binary_decision_setupOLD(self):
        self.TRI.bd_target_alpha = Neuron.output_neuron.activation.bd_defaults[0]
        self.TRI.bd_target_beta  = Neuron.output_neuron.activation.bd_defaults[1]
        self.TRI.bd_threshold    = Neuron.output_neuron.activation.bd_defaults[2]
        #self.TRI.bd_label_alpha = self.TRI.training_data.target_labels[0]
        #self.TRI.bd_label_beta  = self.TRI.training_data.target_labels[1]
        self.TRI.bd_target_alpha_unscaled = self.TRI.training_data.target_min
        self.TRI.bd_target_beta_unscaled  = self.TRI.training_data.target_max

    def binary_decision_setup(self):
        # Set binary decision targets from actual training data
        self.TRI.bd_target_alpha_unscaled = self.TRI.training_data.target_min
        self.TRI.bd_target_beta_unscaled = self.TRI.training_data.target_max

        # Calculate threshold as midpoint of actual targets
        self.TRI.bd_threshold = (self.TRI.bd_target_alpha_unscaled + self.TRI.bd_target_beta_unscaled) / 2.0

        # For now, set scaled versions same as unscaled (threshold logic doesn't use them)
        self.TRI.bd_target_alpha = self.TRI.bd_target_alpha_unscaled
        self.TRI.bd_target_beta = self.TRI.bd_target_beta_unscaled


    def configure_model(self, config: Config):   pass  # Typically overwritten in child  class.

    def customize_neurons(self, config: Config): pass  # Typically overwritten in child  class.

    def scale_samples(self):  # Scales the inputs and targets according to the config in the model and config defaults
        self.config.scaler.scale_all()

    def train(self, exploratory_epochs = 0):
        """
        Main method invoked from Framework to train model.

        Parameters:
            exploratory_epochs: If doing a LR sweep or something, how many epochs to check.
        Returns:
            None
        """
        self.config.loss_function.validate_activation_functions()
        self.scale_samples()
        epochs_to_run = self.number_of_epochs if exploratory_epochs == 0 else exploratory_epochs

        for epoch in range(epochs_to_run):                               # Loop to run specified # of epochs
            if self.should_print_epoch(epoch,exploratory_epochs): print(f"Epoch: {epoch} for {self.TRI.gladiator} MAE = {self.TRI.mae} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.TRI.converge_cond = self.run_an_epoch(epoch)            # Call function to run single epoch
            if self.TRI.converge_cond != "Did Not Converge":   return    # Converged so end early


    def run_an_epoch(self, epoch: int) -> str:
        """
        Executes a training epoch i.e. trains on all samples

        Args:
            epoch (int): number of epoch being executed
        Returns:
            convergence_signal (str): If not converged, empty string, otherwise signal that detected convergence
        """
        self.epoch = epoch  # Set so the child model has access

        for self.iteration, (sample, sample_unscaled) in enumerate(
            zip(self.config.scaler.scaled_samples, self.config.scaler.unscaled_samples)
        ):
            if self.run_a_sample(sample, sample_unscaled) == "Gradient Explosion":
                return "Gradient Explosion"

            if self.VCR.abs_error_for_epoch > 1e21:
                self.VCR.finish_epoch(epoch + 1)
                return "Gradient Explosion"  # Check for gradient explosion
        return self.VCR.finish_epoch(epoch + 1)  # Finish epoch and return convergence signal

    def threshold(self, value):
        if value >= self.TRI.bd_threshold:
            return self.TRI.bd_target_beta
        else:
            return self.TRI.bd_target_alpha

    def should_print_epoch(self, epoch,exp):
        if exp>0:                   return False
        if epoch == 0:              return False            # preserve old "epoch != 0" behavior
        if epoch <= 50:             return True             # always print 1–10
        if epoch <= 100:            return epoch % 10 == 0  # 20, 30, 40, ... 100
        return epoch % 100 == 0                             # back to "normal" over 100

    def run_a_sample(self, sample, sample_unscaled):
        self                .snapshot_weights("", "_before")
        error, loss, blame  = self.optimize_passes(sample)
        if error == "Gradient Explosion": return error

        # 4 possible prediction values...   prediction_raw, prediction_raw_unscaled,
        # prediction_thresholded, prediction_thresholded_unscaled(called just prediction) for binary decision
        prediction_raw = Neuron.output_neuron.activation_value
        prediction_unscaled = self.config.scaler.unscale_target(prediction_raw)
        prediction_thresh = self.threshold(prediction_unscaled)  # Use unscaled value!

        # Step 4: Record iteration data
        iteration_data = Iteration(
            run_id              = self.TRI.run_id,
            epoch               = self.epoch + 1,
            iteration           = self.iteration + 1,
            inputs              = dumps(sample[:-1]),
            inputs_unscaled     = dumps(sample_unscaled[:-1]),
            target              = sample[-1],
            target_unscaled     = sample_unscaled[-1],
            prediction          = prediction_thresh,
            prediction_unscaled = self.config.scaler.unscale_target(prediction_raw),
            prediction_raw      = prediction_raw,
            loss                = loss,
            loss_function       = self.config.loss_function.name,
            loss_gradient       = blame,
            accuracy_threshold  = 1e-10,
        )
        self.VCR.record_iteration(iteration_data, Neuron.layers)

    def optimize_passes(self, sample_scaled):
        # Step 1: Forward pass
        prediction_raw = self.forward_pass(sample_scaled)
        if prediction_raw > 1e69: return "Gradient Explosion", None, None
        if prediction_raw is None:
            raise ValueError(f"{self.__class__.__name__}.forward_pass must return a value for sample={sample!r}")

        # Step 2: Judge pass - calculate error, loss, and blame (gradient)
        error_scaled, loss, loss_gradient = self.judge_pass(sample_scaled, prediction_raw)

        # Step 3: Backwards Pass
        self.back_pass(sample_scaled, loss_gradient)

        # 🎯 Record blame and weight updates for NeuroForge
        self.VCR.record_blame_calculations(self.blame_calculations)
        self.VCR.record_weight_updates(self.weight_calculations, "update")
        return error_scaled, loss, loss_gradient

    ################################################################################################
    ################################ SECTION 1 - Training Default Methods ##########################
    ################################################################################################
    def forward_pass(self, training_sample: Tuple[float, float, float]) -> float:
        """
        🚀 Computes forward pass for each neuron in the XOR MLP.
        🔍 Activation of Output neuron will be considered 'Raw Prediction'
        """
        input_values = training_sample[:-1]
        for layer_idx, layer in enumerate(Neuron.layers):
            prev_activations = (
                input_values if layer_idx == 0
                else [n.activation_value for n in Neuron.layers[layer_idx - 1]]
            )
            for neuron in layer:
                neuron.raw_sum = sum(input_val * weight for input_val, weight in zip(prev_activations, neuron.weights))
                neuron.raw_sum += neuron.bias
                if neuron.raw_sum > 1e69:
                    return neuron.raw_sum  # Catch gradient explosion
                neuron.activate()
        return Neuron.output_neuron.activation_value

    def back_pass(self, training_sample: Tuple[float, float, float], loss_gradient: float):
        """
        # Step 1: Compute blame for output neuron
        # Step 2: Compute blame for hidden neurons
        # Step 3: Adjust weights (Spread the blame)
        """
        self.back_pass__determine_blame_for_output_neuron(loss_gradient)
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):
            for hidden_neuron in Neuron.layers[layer_index]:
                self.back_pass__determine_blame_for_a_hidden_neuron(hidden_neuron)
        self.back_pass__spread_the_blame(training_sample)

    def back_pass__determine_blame_for_output_neuron(self, loss_gradient: float):
        activation_gradient               = Neuron.output_neuron.activation_gradient
        blame                             = loss_gradient * activation_gradient
        Neuron.output_neuron.error_signal = blame

    def back_pass__determine_blame_for_a_hidden_neuron(self, neuron: Neuron):
        activation_gradient      = neuron.activation_gradient
        total_backprop_error     = 0
        for next_neuron in Neuron.layers[neuron.layer_id + 1]:
            weight_to_next       = next_neuron.weights_before[neuron.position]
            error_from_next      = next_neuron.error_signal
            total_backprop_error += weight_to_next * error_from_next
            neuron.error_signal   = activation_gradient * total_backprop_error
            self.blame_calculations.append([
                self.epoch+1, self.iteration+1, self.TRI.run_id,
                neuron.nid, next_neuron.position,
                weight_to_next, "*", error_from_next,
                "=", None, None,
                weight_to_next * error_from_next
            ])

    def back_pass__spread_the_blame(self, training_sample: Tuple[float, float, float]):
        for layer_index in range(len(Neuron.layers) - 1, -1, -1):
            prev_layer = (
                training_sample[:-1] if layer_index == 0
                else [n.activation_value for n in Neuron.layers[layer_index - 1]]
            )
            for neuron in Neuron.layers[layer_index]:
                self.back_pass__update_neurons_weights(neuron, prev_layer)

    def back_pass__update_neurons_weights(self, neuron: Neuron, prev_layer_values: List[float]) -> None:
        blame        = neuron.error_signal
        input_vector = [1.0] + list(prev_layer_values)
        self.weight_calculations.extend(
            self.config.optimizer.update(
                neuron, input_vector, blame, self.total_iterations,
                config    = self.config,
                epoch     = self.epoch + 1,
                iteration = self.iteration + 1,
                batch_id  = self.VCR.batch_id
            )
        )
        self.total_iterations += len(input_vector)

    ############################### END OF BACKPASS ###############################
    ############################### END OF BACKPASS ###############################
    ############################### END OF BACKPASS ###############################

    def judge_pass(self, sample, prediction_raw: float):
        """
        Computes error, loss, and blame based on the configured loss function.
        """
        target  = sample[-1]
        error   = target - prediction_raw
        loss    = self.config.loss_function(prediction_raw, target)
        blame   = self.config.loss_function.grad(prediction_raw, target)
        return error, loss, blame

    ################################################################################################
    ################################ SECTION 3 - Initialization ####################################
    ################################################################################################

    def snapshot_weights(self, from_suffix: str, to_suffix: str):
        """
        Copies weights and biases from one named attribute to another for all neurons.
        """
        for layer in Neuron.layers:
            for neuron in layer:
                from_weights = getattr(neuron, f"weights{from_suffix}")
                from_bias    = getattr(neuron, f"bias{from_suffix}")

                setattr(neuron, f"weights{to_suffix}", from_weights.copy())
                setattr(neuron, f"bias{to_suffix}",    from_bias)

    def update_best_weights_if_new_lowest_errorDELETEME(self, current_error: float):
        """
        Checks if the current error is the lowest seen so far, and if so,
        stores weights and bias as the new best.
        """
        if not hasattr(self, 'lowest_error') or current_error < self.lowest_error:
            self.lowest_error = current_error
            self.snapshot_weights("", "_best")

    def restore_best_weights_from_run(self):
        """
        Restores the best weights and biases recorded during the run.
        Assumes snapshot_weights("_best", "") was used to store them.
        """
        if not hasattr(self, 'lowest_error'):
            print("⚠️ No best weights found — 'lowest_error' was never set.")
            return

    def get_flat_initializers(self, architecture: List[int], initializers: List[WeightInitializer]) -> List[WeightInitializer]:
        """
        Returns a flat list of weight initializers, ensuring compatibility with the number of neurons.
        """
        total_neurons = sum(architecture)
        if not initializers:
            initializers = [Initializer_Xavier] * total_neurons
        if len(initializers) == total_neurons:
            return initializers
        if len(initializers) == len(architecture):
            expanded_list = []
            for layer_index, neuron_count in enumerate(architecture):
                expanded_list.extend([initializers[layer_index]] * neuron_count)
            return expanded_list
        if len(initializers) == 1:
            return [initializers[0]] * total_neurons
        raise ValueError(
            f"Incompatible number of initializers ({len(initializers)}) "
            f"for total neurons ({total_neurons})"
        )

    def initialize_neurons(self,
                           architecture: List[int],
                           initializers:  List[WeightInitializer] = None,
                           hidden_activation: StrategyActivationFunction = None,
                           output_activation: StrategyActivationFunction = None):
        """
        Initializes neurons based on the specified architecture, using appropriate weight initializers.
        """
        if architecture is None:
            raise ValueError("Blueprint is empty - this shouldn't be possible - defect alert")

        flat_initializers = self.get_flat_initializers(architecture, initializers)
        input_count        = self.training_data.input_count
        hidden_activation = hidden_activation or self.config.hidden_activation
        output_activation = output_activation or self.config.loss_function.recommended_output_activation
        nid               = -1

        Neuron.neurons.clear()
        Neuron.layers.clear()

        for layer_index, layer_size in enumerate(architecture):
            if layer_index == 0:
                num_of_weights = self.training_data.input_count
            else:
                num_of_weights = architecture[layer_index - 1]

            for neuron_index in range(layer_size):
                nid += 1
                activation = (
                    output_activation
                    if layer_index == len(architecture) - 1
                    else hidden_activation
                )
                Neuron(
                    nid                = nid,
                    num_of_weights     = num_of_weights,
                    learning_rate      = self.config.learning_rate,
                    weight_initializer = flat_initializers[nid],
                    layer_id           = layer_index,
                    activation         = activation
                )

    ################################################################################################
    ################################ SECTION 4 - Misc ##############################################
    ################################################################################################

    def on_epoch_end(self, epoch, error_summary):
        """
        Optional override: Called after each epoch.
        """
        pass

    def on_training_complete(self):
        """
        Optional override: Called after training run completes.
        """
        pass