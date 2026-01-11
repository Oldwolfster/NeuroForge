# pylint: disable=no-self-use
from abc                       import ABC
from json                      import dumps
from src.NNA.legos.Activation import *
from src.NNA.engine.RecordSample import RecordSample
from src.NNA.engine.VCRRecorder import VCR
from src.NNA.utils.general_utils import *
from src.NNA.legos.Optimizer import *
from src.NNA.engine.BinaryDecision import BinaryDecision
from src.NNA.engine.Config import Config
from src.NNA.engine.TrainingRunInfo import TrainingRunInfo
#from src.NNA.engine.VCRRecorder          import VCR
#from src.NNA.engine.Config       import Config
from src.NNA.engine.Neuron       import Neuron
from datetime                    import datetime

from src.NNA.utils.snapshot_weights import snapshot_weights


#from src.NNA.engine.Utils_DataClasses import sample
#from src.NNA.legos.WeightInitializers import *



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
        snapshot_weights                ("", "_before")
        error, loss, blame              = self.run_passes(sample)
        if error                        == "Gradient Explosion":       return error
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

        self.config.optimizer.optimize_sample(sample_results,self.TRI)

        self.VCR.record_blame_calculations(self.blame_calculations)
        #self.VCR.record_weight_updates(self.weight_calculations, "update")
        self.VCR.record_sample(sample_results, Neuron.layers)

    def run_passes(self, sample_scaled):
        prediction_raw = self.forward_pass(sample_scaled)
        error_scaled, loss, loss_gradient = self.judge_pass(sample_scaled, prediction_raw)
        self.back_pass(loss_gradient)

        if self.has_gradient_explosion():         return "Gradient Explosion", None, None

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
                neuron.neuron_inputs = neuron_inputs
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

    def back_pass(self, loss_gradient: float):
        """
        Single pass through all neurons (right to left):
        1. Determine blame (output vs hidden logic)
        2. Accumulate leverage for weight updates
        """
        for layer_index in range(len(Neuron.layers) - 1, -1, -1):
            for neuron in Neuron.layers[layer_index]:
                self.back_pass__determine_blame(neuron, loss_gradient)
                self.back_pass__calculate_and_accumulate_leverage(neuron)

    def back_pass__determine_blame(self, neuron: Neuron, loss_gradient: float):
        """Dispatch to appropriate blame calculation based on neuron type."""
        if neuron == Neuron.output_neuron:  self.back_pass__determine_blame_for_output_neuron(neuron, loss_gradient)
        else:                               self.back_pass__determine_blame_for_hidden_neuron(neuron)

    def back_pass__determine_blame_for_output_neuron(self, neuron: Neuron, loss_gradient: float):
        neuron.accepted_blame = loss_gradient * neuron.activation_gradient

    def back_pass__determine_blame_for_hidden_neuron(self, neuron: Neuron):
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

    def back_pass__calculate_and_accumulate_leverage(self, neuron: Neuron):
        """Calculate and accumulate leverage for each weight."""
        for weight_id, input_value in enumerate(neuron.neuron_inputs):
            leverage = input_value * neuron.accepted_blame
            neuron.accumulated_leverage[weight_id] += leverage