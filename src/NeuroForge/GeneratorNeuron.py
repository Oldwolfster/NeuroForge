# GeneratorNeuron.py

import copy
from src.NeuroForge import Const
from src.NeuroForge.DisplayModel__Neuron import DisplayModel__Neuron
from src.NeuroForge.DisplayModel__NeuronScaler import DisplayModel__NeuronScaler
from src.NeuroForge.DisplayModel__Layer import DisplayModel__Layer


class GeneratorNeuron:
    model = None
    nid = 0
    output_layer = 0

    @staticmethod
    def create_neurons(the_model, max_act: float,
                       margin=Const.NEURON_MARGIN,
                       min_gap=Const.NEURON_MIN_GAP,
                       max_neuron_size=Const.NEURON_MAX_SIZE):
        """Create neuron objects, dynamically positioning them based on architecture."""
        GeneratorNeuron.nid = 0
        GeneratorNeuron.model = the_model
        GeneratorNeuron.model.layers = []
        GeneratorNeuron.model.neurons = []

        available_height = GeneratorNeuron.model.height
        available_width = GeneratorNeuron.model.width
        full_architecture = GeneratorNeuron.get_full_architecture()
        print(f"the model= {the_model.TRI.gladiator} config architecture ={the_model.TRI.config.architecture} - full architecture: ", full_architecture)
        layer_count = len(full_architecture)
        layer_width, gap_width = GeneratorNeuron.calculate_column_width(
            layer_count, max_neuron_size, min_gap, available_width
        )

        true_layer_index = 0

        for layer_index, neuron_count in enumerate(full_architecture):
            x_position = layer_index * layer_width + ((layer_index + 1) * gap_width)

            if neuron_count > 0:
                layer_obj = DisplayModel__Layer(
                    model=GeneratorNeuron.model,
                    layer_index=true_layer_index,
                    x_position=x_position,
                    width=layer_width,
                    available_height=available_height
                )
                GeneratorNeuron.model.layers.append(layer_obj)

            GeneratorNeuron.create_layer(
                x_position, layer_width, layer_index, neuron_count,
                max_neuron_size, min_gap, available_height, max_act, true_layer_index
            )

            if neuron_count > 0:
                true_layer_index += 1

        GeneratorNeuron.separate_graph_holder_from_neurons()
        GeneratorNeuron.model.layer_width = layer_width

    @staticmethod
    def get_full_architecture():
        """Build full architecture including scaler pseudo-layers."""
        GeneratorNeuron.output_layer = len(GeneratorNeuron.model.config.architecture) - 1
        full_architecture = copy.deepcopy(GeneratorNeuron.model.config.architecture)

        # Ensure output layer has room for graph
        if full_architecture and full_architecture[-1] < 2:
            full_architecture[-1] = 2

        # Prepend scaler if inputs are scaled
        if GeneratorNeuron.model.config.scaler.inputs_are_scaled:
            full_architecture.insert(0, -1)
            GeneratorNeuron.output_layer += 1

        # Append prediction scaler
        full_architecture.append(-1)
        return full_architecture

    @staticmethod
    def calculate_column_width(layer_count, max_neuron_size, min_gap, available_width):
        """Calculate optimal layer width and gap to fill available space."""
        gap_count = layer_count + 1
        entire_width = layer_count * max_neuron_size + (min_gap * gap_count)
        layer_width = max_neuron_size
        gap_width = min_gap

        if entire_width > available_width:
            overrun = entire_width - available_width
            layer_width -= overrun / layer_count
        else:
            extra = available_width - entire_width
            gap_width += extra / gap_count

        return layer_width, gap_width

    @staticmethod
    def create_layer(x_position, layer_width, layer_index, neuron_count,
                     max_neuron_size, min_gap, available_height, max_act, true_layer_index):
        """Create neurons for a single layer."""
        neuron_height, gap_height = GeneratorNeuron.calculate_column_width(
            abs(neuron_count), max_neuron_size, min_gap, available_height
        )
        text_version = "Concise" if layer_width < 350 else "Verbose"

        if neuron_count > 0:
            GeneratorNeuron.create_regular_neurons(
                x_position, layer_width, neuron_height, gap_height,
                neuron_count, true_layer_index, text_version, max_act
            )
        else:
            GeneratorNeuron.create_scaler_neuron(
                x_position, layer_width, neuron_height, gap_height,
                layer_index, text_version, max_act
            )

    @staticmethod
    def create_regular_neurons(x_position, layer_width, neuron_height, gap_height,
                               neuron_count, true_layer_index, text_version, max_act):
        """Create standard neurons for a layer."""
        layer_neurons = []
        current_layer = GeneratorNeuron.model.layers[-1] if GeneratorNeuron.model.layers else None

        for neuron_index in range(neuron_count):
            y_position = neuron_index * neuron_height + ((neuron_index + 1) * gap_height)

            neuron = DisplayModel__Neuron(
                GeneratorNeuron.model,
                left=x_position,
                top=y_position,
                width=layer_width,
                height=neuron_height,
                nid=GeneratorNeuron.nid,
                layer=true_layer_index,
                position=neuron_index,
                output_layer=GeneratorNeuron.output_layer,
                text_version=text_version,
                run_id=GeneratorNeuron.model.run_id,
                screen=GeneratorNeuron.model.surface,
                max_activation=max_act
            )

            if current_layer:
                current_layer.add_neuron(neuron)

            layer_neurons.append(neuron)
            GeneratorNeuron.nid += 1

        GeneratorNeuron.model.neurons.append(layer_neurons)

        if current_layer:
            current_layer.determine_control_needs()

    @staticmethod
    def create_scaler_neuron(x_position, layer_width, neuron_height, gap_height,
                             layer_index, text_version, max_act):
        """Create input or prediction scaler neuron."""
        y_position = gap_height
        is_input = (layer_index == 0)

        scaler_neuron = DisplayModel__NeuronScaler(
            left=x_position,
            top=y_position,
            width=layer_width,
            height=neuron_height,
            nid=-1,
            layer=-1,
            position=0,
            output_layer=0,
            text_version=text_version,
            run_id=GeneratorNeuron.model.run_id,
            model=GeneratorNeuron.model,
            screen=GeneratorNeuron.model.surface,
            max_activation=max_act,
            is_input=is_input
        )

        if is_input:
            GeneratorNeuron.model.input_scaler_neuron = scaler_neuron
        else:
            GeneratorNeuron.model.prediction_scaler_neuron = scaler_neuron
            if GeneratorNeuron.model.TRI.training_data.is_binary_decision:
                GeneratorNeuron.create_thresholder(
                    x_position, y_position, layer_width, neuron_height, text_version, max_act
                )

    @staticmethod
    def create_thresholder(x_position, y_position, layer_width, neuron_height, text_version, max_act):
        """Create the binary decision thresholder visualization."""
        GeneratorNeuron.model.thresholder = DisplayModel__NeuronScaler(
            left=x_position,
            top=y_position,
            width=layer_width,
            height=neuron_height,
            nid=-2,
            layer=-1,
            position=0,
            output_layer=0,
            text_version=text_version,
            run_id=GeneratorNeuron.model.run_id,
            model=GeneratorNeuron.model,
            screen=GeneratorNeuron.model.surface,
            max_activation=max_act,
            is_input=False
        )

    @staticmethod
    def separate_graph_holder_from_neurons():
        """Remove the last neuron slot to use as graph placeholder."""
        if GeneratorNeuron.model.neurons and GeneratorNeuron.model.neurons[-1]:
            graph_slot = GeneratorNeuron.model.neurons[-1].pop()
            GeneratorNeuron.model.graph_holder = graph_slot

            if GeneratorNeuron.model.layers and GeneratorNeuron.model.layers[-1].neurons:
                if GeneratorNeuron.model.layers[-1].neurons[-1] == graph_slot:
                    GeneratorNeuron.model.layers[-1].neurons.pop()