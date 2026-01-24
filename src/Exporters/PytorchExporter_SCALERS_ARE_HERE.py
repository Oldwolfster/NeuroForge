from datetime import datetime
import os

class PytorchExporter:
    """
    Generates a standalone PyTorch script from a completed NNA training run.
    The script includes embedded training data, scalers, model, and training loop
    with CSV output for comparison against NNA results.
    """

    def __init__(self, TRI):
        self.TRI = TRI
        self.config = TRI.config
        self.db = TRI.db

    def generate(self, script_dir="history/"):
        """Main entry point - generates complete PyTorch script."""

        timestamp = datetime.now().strftime("%y%m%d_%H_%M")
        filename = f"nna_model_pytorch_{timestamp}.py"

        os.makedirs(script_dir, exist_ok=True)
        filepath = os.path.join(script_dir, filename)
        self.script_dir = os.path.abspath(script_dir)

        sections = [
            self.build_header(),
            self.build_training_data(),
            self.build_scalers(),
            self.build_model_class(),
            self.build_training_loop(),
            self.build_main(),
        ]
        script = "\n\n".join(sections)

        with open(filepath, "w") as f:
            f.write(script)

        print(f"PyTorch script exported to: {filepath}")
        return filepath

    def build_header(self):
        """Generate imports and metadata comment."""
        run_id = self.TRI.run_id
        lines = [
            "# Auto-generated PyTorch script from NeuroForge",
            f"# Run ID: {run_id}",
            "# Generated for validation against NNA implementation",
            "",
            "import torch",
            "import torch.nn as nn",
            "import torch.optim as optim",
            "import csv",
            "from datetime import datetime",
        ]
        return "\n".join(lines)

    def build_training_data(self):
        """Embed raw training data as Python lists."""
        raw_data = self.TRI.training_data.raw_data

        X_data = [list(row[:-1]) for row in raw_data]
        Y_data = [row[-1] for row in raw_data]

        lines = [
            "# Training Data (raw, before scaling)",
            f"X_RAW = {X_data}",
            f"Y_RAW = {Y_data}",
        ]
        return "\n".join(lines)

    def build_scalers(self):
        """Generate scaler classes with fitted parameters."""
        lines = [
            "# Scalers with fitted parameters from NNA",
            "",
            self.build_scaler_base_class(),
            "",
            self.build_input_scalers(),
            "",
            self.build_target_scaler(),
            "",
            self.build_scale_data_function(),
        ]
        return "\n".join(lines)

    def build_scaler_base_class(self):
        lines = [
            "class Scaler:",
            '    """Base scaler with fitted parameters."""',
            "    def __init__(self, params):",
            "        self.params = params",
            "    ",
            "    def scale(self, values):",
            "        raise NotImplementedError",
            "    ",
            "    def unscale(self, values):",
            "        raise NotImplementedError",
            "",
            "",
            "class NoScaler(Scaler):",
            "    def scale(self, values):",
            "        return values",
            "    ",
            "    def unscale(self, values):",
            "        return values",
            "",
            "",
            "class MinMaxScaler(Scaler):",
            "    def scale(self, values):",
            '        min_val, max_val = self.params["min"], self.params["max"]',
            "        range_val = max_val - min_val",
            "        if range_val == 0:",
            "            return [0.0 for _ in values]",
            "        return [(v - min_val) / range_val for v in values]",
            "    ",
            "    def unscale(self, values):",
            '        min_val, max_val = self.params["min"], self.params["max"]',
            "        return [v * (max_val - min_val) + min_val for v in values]",
            "",
            "",
            "class MinMaxNeg1to1Scaler(Scaler):",
            "    def scale(self, values):",
            '        min_val, max_val = self.params["min"], self.params["max"]',
            "        range_val = max_val - min_val",
            "        if range_val == 0:",
            "            return [0.0 for _ in values]",
            "        return [((v - min_val) / range_val) * 2 - 1 for v in values]",
            "    ",
            "    def unscale(self, values):",
            '        min_val, max_val = self.params["min"], self.params["max"]',
            "        return [((v + 1) / 2) * (max_val - min_val) + min_val for v in values]",
            "",
            "",
            "class ZScoreScaler(Scaler):",
            "    def scale(self, values):",
            '        mean, std = self.params["mean"], self.params["std"]',
            "        return [(v - mean) / std for v in values]",
            "    ",
            "    def unscale(self, values):",
            '        mean, std = self.params["mean"], self.params["std"]',
            "        return [v * std + mean for v in values]",
            "",
            "",
            "class RobustScaler(Scaler):",
            "    def scale(self, values):",
            '        median, iqr = self.params["median"], self.params["iqr"]',
            "        return [(v - median) / iqr for v in values]",
            "    ",
            "    def unscale(self, values):",
            '        median, iqr = self.params["median"], self.params["iqr"]',
            "        return [v * iqr + median for v in values]",
        ]
        return "\n".join(lines)

    def build_input_scalers(self):
        """Generate input scaler instances with fitted params."""
        lines = [
            "# Input scalers (one per feature)",
            "INPUT_SCALERS = [",
        ]

        multi = self.config.scaler
        for i in range(self.TRI.training_data.input_count):
            scaler = multi.scalers[i]
            scaler_class = self.map_scaler_class(scaler.name)
            params = dict(scaler.params)
            lines.append(f"    {scaler_class}({params}),")

        lines.append("]")
        return "\n".join(lines)

    def build_target_scaler(self):
        """Generate target scaler instance with fitted params."""
        multi = self.config.scaler
        scaler = multi.scalers[-1]
        scaler_class = self.map_scaler_class(scaler.name)
        params = dict(scaler.params)

        return f"TARGET_SCALER = {scaler_class}({params})"

    def map_scaler_class(self, name):
        """Map NNA scaler name to generated class name."""
        mapping = {
            "No Scaling": "NoScaler",
            "Min-Max": "MinMaxScaler",
            "Min-Max-Zero-Centered": "MinMaxNeg1to1Scaler",
            "Z-Score": "ZScoreScaler",
            "Robust": "RobustScaler",
        }
        return mapping.get(name, "NoScaler")

    def build_scale_data_function(self):
        """Generate function to apply scaling to data."""
        lines = [
            "def scale_data(X_raw, Y_raw):",
            '    """Apply fitted scalers to raw data."""',
            "    X_scaled = []",
            "    for row in X_raw:",
            "        scaled_row = []",
            "        for i, val in enumerate(row):",
            "            scaled_row.append(INPUT_SCALERS[i].scale([val])[0])",
            "        X_scaled.append(scaled_row)",
            "    ",
            "    Y_scaled = TARGET_SCALER.scale(Y_raw)",
            "    ",
            "    return X_scaled, Y_scaled",
        ]
        return "\n".join(lines)

    def build_model_class(self):
        """Generate PyTorch nn.Module matching NNA architecture."""
        arch = self.config.architecture
        input_count = self.TRI.training_data.input_count

        hidden_act = self.map_activation(self.config.hidden_activation.name)
        output_act = self.map_activation(self.config.output_activation.name)

        initial_weights = self.get_initial_weights()

        lines = [
            "# Model Definition",
            f"INPUT_COUNT = {input_count}",
            f"ARCHITECTURE = {arch}  # hidden layers + output",
            "",
            self.build_initial_weights_constant(initial_weights),
            "",
            "class NNAModel(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        ",
            "        layer_sizes = [INPUT_COUNT] + ARCHITECTURE",
            "        self.layers = nn.ModuleList()",
            "        ",
            "        for i in range(len(layer_sizes) - 1):",
            "            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))",
            "        ",
            f"        self.hidden_activation = {hidden_act}",
            f"        self.output_activation = {output_act}",
            "        ",
            "        self.init_weights_from_nna()",
            "    ",
            "    def init_weights_from_nna(self):",
            '        """Initialize weights to match NNA starting weights."""',
            "        for layer_idx, layer in enumerate(self.layers):",
            "            with torch.no_grad():",
            "                for neuron_idx in range(layer.out_features):",
            "                    key = (layer_idx, neuron_idx)",
            "                    if key in INITIAL_WEIGHTS:",
            "                        weights = INITIAL_WEIGHTS[key]",
            "                        # weights[0] is bias, weights[1:] are input weights",
            "                        layer.bias[neuron_idx] = weights[0]",
            "                        layer.weight[neuron_idx] = torch.tensor(weights[1:], dtype=torch.float32)",
            "    ",
            "    def forward(self, x):",
            "        for i, layer in enumerate(self.layers[:-1]):",
            "            x = layer(x)",
            "            x = self.hidden_activation(x)",
            "        ",
            "        x = self.layers[-1](x)",
            "        if self.output_activation is not None:",
            "            x = self.output_activation(x)",
            "        ",
            "        return x",
        ]
        return "\n".join(lines)

    def build_initial_weights_constant(self, weights_dict):
        """Format initial weights as Python dict constant."""
        lines = [
            "# Initial weights from NNA: {(layer_idx, neuron_idx): [bias, w0, w1, ...]}",
            "INITIAL_WEIGHTS = {",
        ]

        for key in sorted(weights_dict.keys()):
            weights = weights_dict[key]
            weights_str = ", ".join(f"{w:.10f}" for w in weights)
            lines.append(f"    {key}: [{weights_str}],")

        lines.append("}")
        return "\n".join(lines)

    def map_activation(self, name):
        """Map NNA activation name to PyTorch activation."""
        mapping = {
            "Sigmoid": "nn.Sigmoid()",
            "Tanh": "nn.Tanh()",
            "ReLU": "nn.ReLU()",
            "LeakyReLU": "nn.LeakyReLU(0.01)",
            "Leaky": "nn.LeakyReLU(0.01)",
            "NoneDummmy": "None",
        }
        return mapping.get(name, "None")

    def get_initial_weights(self):
        """Query DB for weights at epoch 1, sample 1 (before any updates)."""
        sql = """
            SELECT nid, weight_id, value_before
            FROM Weight
            WHERE run_id = ? AND epoch = 1 AND sample = 1
            ORDER BY nid, weight_id
        """
        rows = self.db.query(sql, (self.TRI.run_id,))

        # Group by neuron
        neuron_weights = {}
        for nid, weight_id, value in rows:
            if nid not in neuron_weights:
                neuron_weights[nid] = {}
            neuron_weights[nid][weight_id] = value

        # Convert nid to (layer_idx, neuron_idx) and flatten weights
        result = {}
        arch = self.config.architecture
        input_count = self.TRI.training_data.input_count

        nid = 0
        for layer_idx, layer_size in enumerate(arch):
            for neuron_idx in range(layer_size):
                if nid in neuron_weights:
                    weights_dict = neuron_weights[nid]
                    # Flatten: weight_id 0 is bias, 1..n are input weights
                    max_wid = max(weights_dict.keys())
                    weights = [weights_dict.get(i, 0.0) for i in range(max_wid + 1)]
                    result[(layer_idx, neuron_idx)] = weights
                nid += 1

        return result

    def build_training_loop(self):
        """Generate training loop with CSV output."""
        epochs = self.TRI.hyper.epochs_to_run
        lr = self.config.learning_rate
        batch_size = self.config.batch_size

        loss_fn = self.map_loss(self.config.loss_function.name)
        optimizer_code = self.map_optimizer(self.config.optimizer.name, lr)

        lines = [
            "# Training Configuration",
            f"EPOCHS = {epochs}",
            f"LEARNING_RATE = {lr}",
            f"BATCH_SIZE = {batch_size}",
            #f'COMPARISON_DB = r"{self.comparison_db}"' if self.comparison_db else 'COMPARISON_DB = None',
            #'COMPARISON_DB = "NF_history.db"  # Same folder as this script',
            f'COMPARISON_DB = r"{os.path.join(self.script_dir, "NF_history.db")}"',
            "",
            "",
            'def train_model(model, X, Y, csv_path="pytorch_weights.csv"):',
            '    """Train model and output weights to CSV after each epoch."""',
            "    ",
            "    X_tensor = torch.tensor(X, dtype=torch.float32)",
            "    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)",
            "    ",
            f"    criterion = {loss_fn}",
            f"    optimizer = {optimizer_code}",
            "    ",
            "    # Open CSV for weight logging",
            '    with open(csv_path, "w", newline="") as csvfile:',
            "        writer = csv.writer(csvfile)",
            '        writer.writerow(["epoch", "layer", "neuron", "weight_index", "weight_value"])',
            "        ",
            "        # Log initial weights (epoch 1, before training - matches NNA convention)",
            "        log_weights(writer, model, epoch=1)",
            "        log_weights_to_db(model, epoch=1)",
            "        ",
            "        for epoch in range(2, EPOCHS + 2):  # Start at 2 since we logged initial as epoch 1",
            "            model.train()",
            "            ",
            "            # Mini-batch training",
            "            indices = list(range(len(X)))",
            "            for start in range(0, len(X), BATCH_SIZE):",
            "                end = min(start + BATCH_SIZE, len(X))",
            "                batch_X = X_tensor[start:end]",
            "                batch_Y = Y_tensor[start:end]",
            "                ",
            "                optimizer.zero_grad()",
            "                outputs = model(batch_X)",
            "                loss = criterion(outputs, batch_Y)",
            "                loss.backward()",
            "                optimizer.step()",
            "            ",
            "            # Log weights after epoch",
            "            log_weights(writer, model, epoch)",
            "            log_weights_to_db(model, epoch)",
            "            ",
            "            # Calculate and print epoch loss",
            "            with torch.no_grad():",
            "                outputs = model(X_tensor)",
            "                epoch_loss = criterion(outputs, Y_tensor).item()",
            '                print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")',
            "    ",
            '    print(f"Weights logged to {csv_path}")',
            "    return model",
            "",
            "",
            "def log_weights(writer, model, epoch):",
            '    """Write all weights to CSV."""',
            "    with torch.no_grad():",
            "        for layer_idx, layer in enumerate(model.layers):",
            "            for neuron_idx in range(layer.out_features):",
            "                # Bias is weight_index 0",
            "                bias_val = layer.bias[neuron_idx].item()",
            "                writer.writerow([epoch, layer_idx, neuron_idx, 0, bias_val])",
            "                ",
            "                # Input weights are weight_index 1, 2, ...",
            "                for w_idx, w_val in enumerate(layer.weight[neuron_idx]):",
            "                    writer.writerow([epoch, layer_idx, neuron_idx, w_idx + 1, w_val.item()])",
        ]
        lines.append("")
        lines.append("")
        lines.append(self.build_db_logging_function())

        return "\n".join(lines)

    def build_db_logging_function(self):
        """Generate function to log weights to SQLite DB."""
        lines = [
            "def log_weights_to_db(model, epoch):",
            '    """Write all weights to SQLite database."""',
            "    if COMPARISON_DB is None:",
            "        return",
            "    ",
            "    import sqlite3",
            #"    conn = sqlite3.connect(COMPARISON_DB)",
            "    import os",
            "    script_dir = os.path.dirname(os.path.abspath(__file__))",
            "    db_path = os.path.join(script_dir, COMPARISON_DB)",
            "    conn = sqlite3.connect(db_path)",


            "    cursor = conn.cursor()",
            "    ",
            "    # Create table if not exists",
            "    cursor.execute('''",
            "        CREATE TABLE IF NOT EXISTS pytorch_weights (",
            "            epoch INTEGER,",
            "            layer INTEGER,",
            "            neuron INTEGER,",
            "            weight_index INTEGER,",
            "            weight_value REAL,",
            "            PRIMARY KEY (epoch, layer, neuron, weight_index)",
            "        )",
            "    ''')",
            "    ",
            "    with torch.no_grad():",
            "        for layer_idx, layer in enumerate(model.layers):",
            "            for neuron_idx in range(layer.out_features):",
            "                # Bias is weight_index 0",
            "                bias_val = layer.bias[neuron_idx].item()",
            "                cursor.execute(",
            '                    "INSERT OR REPLACE INTO pytorch_weights VALUES (?, ?, ?, ?, ?)",',
            "                    (epoch, layer_idx, neuron_idx, 0, bias_val)",
            "                )",
            "                ",
            "                # Input weights are weight_index 1, 2, ...",
            "                for w_idx, w_val in enumerate(layer.weight[neuron_idx]):",
            "                    cursor.execute(",
            '                        "INSERT OR REPLACE INTO pytorch_weights VALUES (?, ?, ?, ?, ?)",',
            "                        (epoch, layer_idx, neuron_idx, w_idx + 1, w_val.item())",
            "                    )",
            "    ",
            "    conn.commit()",
            "    conn.close()",
        ]
        return "\n".join(lines)

    def map_loss(self, name):
        """Map NNA loss name to PyTorch loss."""
        mapping = {
            "Mean Squared Error": "nn.MSELoss()",
            "Mean Absolute Error": "nn.L1Loss()",
            "Binary Cross-Entropy": "nn.BCELoss()",
            "Huber Loss": "nn.HuberLoss()",
        }
        return mapping.get(name, "nn.MSELoss()")

    def map_optimizer(self, name, lr):
        """Map NNA optimizer name to PyTorch optimizer instantiation."""
        beta1 = getattr(self.TRI.hyper, 'optimizer_beta1', 0.9)
        beta2 = getattr(self.TRI.hyper, 'optimizer_beta2', 0.999)
        eps = getattr(self.TRI.hyper, 'optimizer_epsilon', 1e-8)

        mapping = {
            "Stochastic Gradient Descent": f"optim.SGD(model.parameters(), lr={lr})",
            "Adam": f"optim.Adam(model.parameters(), lr={lr}, betas=({beta1}, {beta2}), eps={eps})",
            "AdaMax": f"optim.Adamax(model.parameters(), lr={lr}, betas=({beta1}, {beta2}), eps={eps})",
            "RAdam": f"optim.RAdam(model.parameters(), lr={lr}, betas=({beta1}, {beta2}), eps={eps})",
            "RMSprop": f"optim.RMSprop(model.parameters(), lr={lr})",
            "Adagrad": f"optim.Adagrad(model.parameters(), lr={lr})",
        }
        return mapping.get(name, f"optim.SGD(model.parameters(), lr={lr})")

    def build_main(self):
        """Generate main entry point."""
        lines = [
            'if __name__ == "__main__":',
            '    print("=" * 60)',
            '    print("NeuroForge -> PyTorch Validation Script")',
            '    print("=" * 60)',
            "    ",
            "    # Scale the data",
            "    X_scaled, Y_scaled = scale_data(X_RAW, Y_RAW)",
            "    ",
            '    print(f"Training data: {len(X_scaled)} samples, {len(X_scaled[0])} features")',
            '    print(f"Architecture: {ARCHITECTURE}")',
            '    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")',
            "    print()",
            "    ",
            "    # Create and train model",
            "    model = NNAModel()",
            "    trained_model = train_model(model, X_scaled, Y_scaled)",
            "    ",
            "    print()",
            '    print("Done! Compare pytorch_weights.csv against NNA database.")',
        ]
        return "\n".join(lines) #Thank you Opus