# Auto-generated PyTorch script from NeuroForge
# Run ID: 1
# Generated for validation against NNA implementation

import torch
import torch.nn as nn
import torch.optim as optim
import csv
from datetime import datetime

# Training Data (raw, before scaling)
X_RAW = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_RAW = [0, 1, 1, 0]

# Model Definition
INPUT_COUNT = 2
ARCHITECTURE = [4, 1]  # hidden layers + output

# Initial weights from NNA: {(layer_idx, neuron_idx): [bias, w0, w1, ...]}
INITIAL_WEIGHTS = {
    (0, 0): [0.0000000000, -0.2608089282, -0.5559795115],
    (0, 1): [0.0000000000, -0.9593531778, -1.5459940389],
    (0, 2): [0.0000000000, -1.4826299445, 0.2571239444],
    (0, 3): [0.0000000000, -1.0624230090, 0.8459454336],
    (1, 0): [0.0000000000, -0.7174346289, -1.0632171540, -0.5106611152, -0.1610318276],
}

class NNAModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        layer_sizes = [INPUT_COUNT] + ARCHITECTURE
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        self.hidden_activation = nn.Tanh()
        self.output_activation = nn.Sigmoid()
        
        self.init_weights_from_nna()
    
    def init_weights_from_nna(self):
        """Initialize weights to match NNA starting weights."""
        for layer_idx, layer in enumerate(self.layers):
            with torch.no_grad():
                for neuron_idx in range(layer.out_features):
                    key = (layer_idx, neuron_idx)
                    if key in INITIAL_WEIGHTS:
                        weights = INITIAL_WEIGHTS[key]
                        # weights[0] is bias, weights[1:] are input weights
                        layer.bias[neuron_idx] = weights[0]
                        layer.weight[neuron_idx] = torch.tensor(weights[1:], dtype=torch.float32)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.hidden_activation(x)
        
        x = self.layers[-1](x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x

# Training Configuration
EPOCHS = 262
LEARNING_RATE = 0.5
BATCH_SIZE = 4
COMPARISON_DB = r"C:\SynologyDrive\Development\PyCharm\NeuroForge\history\NF_history.db"


def train_model(model, X, Y, csv_path="pytorch_weights.csv"):
    """Train model and output weights to CSV after each epoch."""
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.999), eps=1e-08)
    
    # Open CSV for weight logging
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "layer", "neuron", "weight_index", "weight_value"])
        
        # Log initial weights (epoch 1, before training - matches NNA convention)
        log_weights(writer, model, epoch=1)
        log_weights_to_db(model, epoch=1)
        
        for epoch in range(2, EPOCHS + 1):  # Start at 2 since we logged initial as epoch 1
            model.train()
            
            # Mini-batch training
            indices = list(range(len(X)))
            for start in range(0, len(X), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(X))
                batch_X = X_tensor[start:end]
                batch_Y = Y_tensor[start:end]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
            
            # Log weights after epoch
            log_weights(writer, model, epoch)
            log_weights_to_db(model, epoch)
            
            # Calculate and print epoch loss
            with torch.no_grad():
                outputs = model(X_tensor)
                epoch_loss = criterion(outputs, Y_tensor).item()
                print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
    
    print(f"Weights logged to {csv_path}")
    return model


def log_weights(writer, model, epoch):
    """Write all weights to CSV."""
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            for neuron_idx in range(layer.out_features):
                # Bias is weight_index 0
                bias_val = layer.bias[neuron_idx].item()
                writer.writerow([epoch, layer_idx, neuron_idx, 0, bias_val])
                
                # Input weights are weight_index 1, 2, ...
                for w_idx, w_val in enumerate(layer.weight[neuron_idx]):
                    writer.writerow([epoch, layer_idx, neuron_idx, w_idx + 1, w_val.item()])


def log_weights_to_db(model, epoch):
    """Write all weights to SQLite database."""
    if COMPARISON_DB is None:
        return
    
    import sqlite3
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, COMPARISON_DB)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS ix_pytorch_weights_epoch_nid_wid
        ON pytorch_weights(epoch, nid, weight_index);
    ''')
    
    with torch.no_grad():
        neuron_base = 0
        for layer_idx, layer in enumerate(model.layers):
            for neuron_idx in range(layer.out_features):
                nid = neuron_base + neuron_idx
                # Bias is weight_index 0
                bias_val = layer.bias[neuron_idx].item()
                cursor.execute(
                    "INSERT OR REPLACE INTO pytorch_weights VALUES (?, ?, ?, ?, ?, ?)",
                    (epoch, layer_idx, neuron_idx, nid, 0, bias_val)
                )
                
                # Input weights are weight_index 1, 2, ...
                for w_idx, w_val in enumerate(layer.weight[neuron_idx]):
                    cursor.execute(
                        "INSERT OR REPLACE INTO pytorch_weights VALUES (?, ?, ?, ?, ?, ?)",
                        (epoch, layer_idx, neuron_idx, nid, w_idx + 1, w_val.item())
                    )
            neuron_base += layer.out_features
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("=" * 60)
    print("NeuroForge -> PyTorch Validation Script")
    print("=" * 60)
    
    # Scale the data
    #X_scaled, Y_scaled = scale_data(X_RAW, Y_RAW)
    X_scaled, Y_scaled = X_RAW, Y_RAW  # Temporarily removed scaler    
    print(f"Training data: {len(X_scaled)} samples, {len(X_scaled[0])} features")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print()
    
    # Create and train model
    model = NNAModel()
    trained_model = train_model(model, X_scaled, Y_scaled)
    
    print()
    print("Done! Compare pytorch_weights.csv against NNA database.")