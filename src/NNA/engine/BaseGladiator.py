# pylint: disable=no-self-use
from abc                       import ABC
from json                      import dumps
from src.NNA.Legos.Activation import *
from src.NNA.engine.RecordSample import RecordSample
from src.NNA.engine.VCRRecorder import VCR
from src.NNA.utils.general_utils import *
from src.NNA.Legos.Optimizer import *
from src.NNA.engine.BinaryDecision import BinaryDecision
from src.NNA.engine.Config import Config
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
#from src.NNA.engine.VCRRecorder          import VCR
#from src.NNA.engine.Config       import Config
from src.NNA.engine.Neuron       import Neuron
from datetime                    import datetime

from src.NNA.utils.snapshot_weights import snapshot_weights


#from src.NNA.engine.Utils_DataClasses import sample
#from src.NNA.Legos.WeightInitializers import *



class Gladiator(ABC):
    """
    💥 NOTE: The gradient is inverted from the traditional way of thinking.
    Abstract base class for creating Gladiators (neural network models).
    Goal: Give child gladiator class as much power as possible, requiring as little responsibility
    as possible, while allowing for overwriting any step of the process.

    There are three main sections:
    1) Initialization - Preps everything for the framework and gladiator
    2) Training Default Methods - (Forward and Backwards Pass) available for free, but overwritable for experimentation.
    3)

    """

    def __init__(self,  TRI: TrainingRunInfo):
        self.TRI                    = TRI  # TrainingRunInfo
        self.config                 = TRI.config
        self.db                     = TRI.db
        self.training_data          = TRI.training_data         # Only needed for sqlMgr ==> self.ramDb = args[3]

        self.VCR                    = VCR(TRI)
        self.sample_id              = 0
        self.epoch                  = 0
        self.total_samples          = 1                         # Timestep for optimizers such as adam
        self.blame_calculations     = []
        self.weight_calculations    = []
        self.configure_everything()

    def configure_model  (self, config: Config): pass  # Typically overwritten in child  class.
    def customize_neurons(self, config: Config): pass  # Typically overwritten in child  class.

    def configure_everything(self):
        self.configure_model(self.config)              # Typically overwritten in child (gladiator) class.
        #ez_debug(arch=self.config.architecture)
        self.config.autoML()
        self.initialize_neurons()
        self.customize_neurons(self.config)            # Typically overwritten in child  class.
        self.config.scaler.scale_all()
        self.config.loss_function.validate_activation_functions()

    def initialize_neurons(self):
        """Initializes neurons based on config.architecture."""
        Neuron.neurons                  .clear()
        Neuron.layers                   .clear()
        architecture                    = self.config.architecture
        nid                             = -1
        for layer_index,layer_size      in enumerate(architecture):
            num_inputs                  = self.training_data.input_count if layer_index == 0 else architecture[layer_index - 1]
            is_output_layer             = layer_index == len(architecture) - 1
            for _ in range(layer_size):
                nid += 1
                Neuron(
                    nid                 = nid,
                    num_inputs          = num_inputs,
                    learning_rate       = self.config.learning_rate,
                    weight_initializer  = self.config.weight_initializer,
                    layer_id            = layer_index,
                    activation          = self.config.output_activation  if is_output_layer else self.config.hidden_activation
                )


    def train(self, exploratory_epochs = 0):
        """
        Main method invoked from Framework to train model.

        Parameters:
            exploratory_epochs: If doing a LR sweep or something, how many epochs to check.
        Returns:
            None
        """
        epochs_to_run = self.TRI.hyper.epochs_to_run if exploratory_epochs == 0 else exploratory_epochs
        #print(f"epoch_to_run{epochs_to_run}")
        for epoch in range(1, epochs_to_run + 1):                       # Loop to run specified # of epochs
            if should_print_epoch(epoch,exploratory_epochs):            print(f"Epoch: {epoch} for {self.TRI.gladiator} MAE = {self.TRI.mae} ({round(self.TRI.accuracy)}%)at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.TRI.converge_cond = self.run_an_epoch(epoch)           # Call function to run single epoch
            if self.TRI.converge_cond != "Did Not Converge":   return   # Converged so end early


    def run_an_epoch(self, epoch: int) -> str:
        """
        Executes a training epoch i.e. trains on all samples

        Args:
            epoch (int): number of epoch being executed
        Returns:
            convergence_signal (str): If not converged, empty string, otherwise signal that detected convergence
        """
        self.epoch = epoch  # Set so the child model has access

        for self.sample_id, (sample, sample_unscaled) in enumerate(zip(self.config.scaler.scaled_samples, self.config.scaler.unscaled_samples), start=1        ):
            if (self.run_a_sample(sample, sample_unscaled) == "Gradient Explosion" or self.VCR.abs_error_for_epoch > 1e21):
                self.VCR.finish_epoch(epoch )
                return "Gradient Explosion"
        return self.VCR.finish_epoch(epoch)  # Finish epoch and return convergence signal

    def run_a_sample(self, sample, sample_unscaled):
        snapshot_weights("", "_before")
        error, loss, blame = self.optimize_passes(sample)
        if error == "Gradient Explosion":
            return error

        prediction_raw                  = Neuron.output_neuron.activation_value
        prediction_unscaled             = self.config.scaler.unscale_target(prediction_raw)
        prediction_thresh, label        = self.TRI.BD.decide(prediction_unscaled)
        is_true                         = self.TRI.BD.is_true(prediction_unscaled, sample_unscaled[-1])
        #print(f"sample unscaled[:-1]{sample_unscaled[:-1]}")

        sample_results = RecordSample(
            run_id              = self.TRI.run_id,
            epoch               = self.epoch,
            sample_id           = self.sample_id,
            inputs              = dumps(sample[:-1]),
            inputs_unscaled     = dumps(sample_unscaled[:-1]),
            is_true             = is_true,
            target              = sample[-1],
            target_unscaled     = sample_unscaled[-1],
            prediction          = prediction_thresh,
            prediction_unscaled = prediction_unscaled,
            prediction_raw      = prediction_raw,
            prediction_label    = label,
            loss                = loss,
            loss_function       = self.config.loss_function.name,
            loss_gradient       = blame,
            accuracy_threshold  = 1e-10,
        )
        self.VCR.record_sample(sample_results, Neuron.layers)

    def optimize_passes(self, sample_scaled):
        prediction_raw = self.forward_pass(sample_scaled)
        if prediction_raw is None:
            raise ValueError(f"{self.__class__.__name__}.forward_pass must return a value for sample={sample_scaled!r}")

        error_scaled, loss, loss_gradient = self.judge_pass(sample_scaled, prediction_raw)
        self.back_pass(sample_scaled, loss_gradient)

        if self.has_gradient_explosion():
            self.blame_calculations.clear()
            self.weight_calculations.clear()
            return "Gradient Explosion", None, None

        self.VCR.record_blame_calculations(self.blame_calculations)
        #self.VCR.record_weight_updates(self.weight_calculations, "update")
        return error_scaled, loss, loss_gradient

    def has_gradient_explosion(self):
        """Check if any neuron value has exploded."""
        import math
        for layer in Neuron.layers:
            for neuron in layer:
                for val in [neuron.accepted_blame, neuron.activation_value] + neuron.weights:
                    if val is None:                        continue
                    if math.isnan(val) or math.isinf(val) or abs(val) > 1e21:   return True
        return False

    def forward_pass(self, sample):
        inputs = sample[:-1]
        for layer_idx, layer in enumerate(Neuron.layers):
            prev_values = inputs if layer_idx == 0 else [n.activation_value for n in Neuron.layers[layer_idx - 1]]
            neuron_inputs = [1.0] + list(prev_values)  # Bias input + actual inputs
            for neuron in layer:
                neuron.raw_sum = sum(w * x for w, x in zip(neuron.weights, neuron_inputs))
                neuron.activate()
        return Neuron.output_neuron.activation_value


    def judge_pass(self, sample, prediction_raw: float):
        """
        Computes error, loss, and blame based on the configured loss function.
        """
        target  = sample[-1]
        error   = target - prediction_raw
        loss    = self.config.loss_function(prediction_raw, target)
        blame   = self.config.loss_function.grad(prediction_raw, target)
        return error, loss, blame


    def back_pass(self, training_sample: list[float], loss_gradient: float):
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
        Neuron.output_neuron.accepted_blame = blame

    def back_pass__determine_blame_for_a_hidden_neuron(self, neuron: Neuron):
        total_backprop_error = 0
        for next_neuron in Neuron.layers[neuron.layer_id + 1]:
            weight_to_next = next_neuron.weights_before[neuron.position + 1]  # +1 to skip bias
            error_from_next = next_neuron.accepted_blame
            total_backprop_error += weight_to_next * error_from_next
            self.blame_calculations.append([
                self.epoch, self.sample_id, self.TRI.run_id,
                neuron.nid, next_neuron.position,
                weight_to_next, "*", error_from_next,
                "=", None, None,
                weight_to_next * error_from_next
            ])
        neuron.accepted_blame = neuron.activation_gradient * total_backprop_error

    def back_pass__spread_the_blame(self, training_sample: list[float]):
        """Loops through Layers (right to left) then each neuron in Layer
           FOR A NEURON abstracts INPUT from 1) other neurons vs 2) sample inputs.
           Passes to back_pass__blame_per_neuron
        """
        for layer_index in range(len(Neuron.layers) - 1, -1, -1):
            prev_layer = (
                training_sample[:-1] if layer_index == 0
                else [n.activation_value for n in Neuron.layers[layer_index - 1]]
            )
            for neuron in Neuron.layers[layer_index]:
                self.back_pass__accumulate_blame_per_neuron(neuron, prev_layer)

    def back_pass__accumulate_blame_per_neuron(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
        blame = neuron.accepted_blame
        input_vector = [1.0] + list(prev_layer_values)

        if len(input_vector) != len(neuron.weights):
            raise ValueError(
                f"Input vector length ({len(input_vector)}) must match weights length ({len(neuron.weights)}). "
                f"(Did you forget the bias input or include an extra input?)"
            )

        # Ensure accumulator exists and is correctly sized (generic, optimizer-agnostic)
        if (not hasattr(neuron, "accumulated_leverage")
                or neuron.accumulated_leverage is None
                or len(neuron.accumulated_leverage) != len(neuron.weights)):
            neuron.accumulated_leverage = [0.0] * len(neuron.weights)

        # Accumulate "blame spread" per weight (leverage), then (for now) apply immediately (batch_size=1 behavior)
        for weight_id, input_value in enumerate(input_vector):
            leverage = input_value * blame
            neuron.accumulated_leverage[weight_id] += leverage


    def back_pass__accumulate_blame_per_neuronOld(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
        blame = neuron.accepted_blame
        input_vector = [1.0] + list(prev_layer_values)
        self.config.optimizer.update(
            neuron, input_vector, blame, self.total_samples,
            config=self.config,
            epoch=self.epoch,  # Already 1-indexed
            sample=self.sample_id,  # Renamed from sample
            batch_id=self.VCR.batch_id
        )

        self.total_samples += len(input_vector)  # TODO THIS IS WRONG

    def back_pass__blame_per_neuronOLD(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
        blame = neuron.accepted_blame
        input_vector = [1.0] + list(prev_layer_values)
        self.weight_calculations.extend(
            self.config.optimizer.update(
                neuron, input_vector, blame, self.total_samples,
                config=self.config,
                epoch=self.epoch,  # Already 1-indexed
                sample=self.sample_id,  # Renamed from sample
                batch_id=self.VCR.batch_id
            )
        )
        self.total_samples += len(input_vector)  #TODO THIS IS WRONG

