import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
import os
import optuna # Import Optuna
from functools import partial # To pass arguments to objective
import datetime # For checkpoint naming

# --- Constants and Configuration ---
TRAIN_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/train.csv'
TEST_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/test.csv'
SUBMISSION_CSV_PATH = r'submission_1.csv'
MODEL_DIR = 'models'  # Directory to save model checkpoints
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_1.pth')  # Path for best model

# Hyperparameters (will be tuned or set via args)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64 # Default for argparse if not using best
DEFAULT_LR = 1e-4       # Default for argparse if not using best
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_N_TRIALS = 50 # Number of Optuna trials

# Best hyperparameters found by Optuna (hardcoded from previous runs)
# Consider loading these dynamically from study results in a real scenario
BEST_LR = 0.0022063150376473166
BEST_BATCH_SIZE = 32

# Feature lists based on notebook analysis
NUMERICAL_COLS = [
    'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Dates', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumDealsPurchases', 'NumWebVisitsMonth', 'AcceptedCmp1',
    'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain'
]
EDU_COL = 'Education'
MARITAL_COL = 'Marital_Status'
TARGET_COL = 'Target'
DATE_COL = 'Dt_Customer'
ID_COL = 'ID'  # ID column for submission file

N_NUM_FEATURES = len(NUMERICAL_COLS) # Should be 23
N_EDU_FEATURES = 5 # Based on notebook encoding
N_MARITAL_FEATURES = 6 # Based on notebook encoding
# TOTAL_FEATURES constant removed as it was unused

# --- Model Definition ---

# Default list for imputation layer initialization (values are trainable)
default_imputation_list = [1.0] * N_NUM_FEATURES

class Imputation_layer(nn.Module):
    """Trainable layer to impute missing values (NaNs) in numerical features."""
    def __init__(self, num_features=N_NUM_FEATURES):
        super(Imputation_layer, self).__init__()
        # Initialize imputation values (e.g., with 1.0) - these will be learned
        initial_values = torch.tensor(default_imputation_list, dtype=torch.float32)
        self.impute = nn.Parameter(initial_values, requires_grad=True)

    def forward(self, x):
        mask = torch.isnan(x)
        # Create a new tensor with the same shape as x
        imputed_x = x.clone()
        # Expand impute values to match batch size for vectorized replacement
        impute_expanded = self.impute.unsqueeze(0).expand(x.size(0), -1)
        # Replace NaNs using the mask and expanded imputation values
        imputed_x[mask] = impute_expanded[mask]
        return imputed_x

class TrainableScaler(nn.Module):
    """Trainable layer to standardize numerical features."""
    def __init__(self, num_features=N_NUM_FEATURES):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(num_features))
        self.std = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        # Add epsilon for numerical stability
        return (x - self.mean) / (self.std + 1e-6)

# Refactored Residual Block
class ConfigurableResidualBlock(nn.Module):
    """Configurable Residual Block with specified activation and dropout."""
    def __init__(self, in_features, out_features, dropout, activation_fn):
        super(ConfigurableResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )

        # Projection shortcut if dimensions don't match
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

        self.activation = activation_fn

    def forward(self, x):
        identity = x
        out = self.block(x)
        identity = self.shortcut(identity)
        out += identity
        out = self.activation(out)
        return out

# Refactored Main Model
class ConfigurableModel(nn.Module):
    """Configurable model architecture accepting hyperparameters."""
    def __init__(self,
                 num_features=N_NUM_FEATURES,
                 edu_features=N_EDU_FEATURES,
                 marital_features=N_MARITAL_FEATURES,
                 embedding_dim=32,
                 layers_list=[64, 128, 256, 512, 256, 128, 64],
                 dropout_small=0.1,
                 dropout_large=0.3,
                 activation_name="SiLU"):
        super(ConfigurableModel, self).__init__()

        self.imputer = Imputation_layer(num_features)
        self.scaler = TrainableScaler(num_features)

        # Select activation function
        if activation_name == "SiLU":
            self.activation = nn.SiLU()
        elif activation_name == "GELU":
            self.activation = nn.GELU()
        elif activation_name == "ReLU":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

        # Enhanced education embedding
        self.edu_linear = nn.Sequential(
            nn.Linear(edu_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.activation,
            nn.Dropout(dropout_small),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.activation,
            nn.Dropout(dropout_small)
        )

        # Enhanced marital status embedding
        self.marital_linear = nn.Sequential(
            nn.Linear(marital_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.activation,
            nn.Dropout(dropout_small),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.activation,
            nn.Dropout(dropout_small)
        )

        # Calculate input size for the main network
        main_input_features = num_features + 2 * embedding_dim

        # Input projection
        self.input_proj = nn.Linear(main_input_features, layers_list[0])
        self.bn_input = nn.BatchNorm1d(layers_list[0])
        self.dropout = nn.Dropout(dropout_large) # Dropout after first projection

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(layers_list) - 1):
            self.res_blocks.append(
                ConfigurableResidualBlock(layers_list[i], layers_list[i+1], dropout_large, self.activation)
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(layers_list[-1], 32), # Project to intermediate size first
            nn.BatchNorm1d(32),
            self.activation,
            nn.Dropout(dropout_small), # Smaller dropout before final layer
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Input x is a list/tuple of tensors: numerical features, education features, marital features
        x_num = x[0]      # Shape: [batch_size, num_features]
        x_edu = x[1]      # Shape: [batch_size, edu_features]
        x_marital = x[2]  # Shape: [batch_size, marital_features]

        # Apply imputation and scaling to numerical features
        x_num = self.imputer(x_num)
        x_num = self.scaler(x_num)

        # Process categorical features through embeddings
        x_edu = self.edu_linear(x_edu)
        x_marital = self.marital_linear(x_marital)

        # Concatenate all features
        x = torch.cat((x_num, x_edu, x_marital), dim=1)

        # Input projection and initial activation/dropout
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Pass through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Output projection
        x = self.output_proj(x)
        return x.squeeze(-1)  # Return predictions: [batch_size]


# --- Data Preprocessing ---

def preprocess_data(df):
    """Applies preprocessing steps from the notebook."""
    df_processed = df.copy()

    # Keep the ID column for later if it exists
    id_values = None
    if ID_COL in df_processed.columns:
        id_values = df_processed[ID_COL].copy()

    # 1. Handle Date Feature ('Dt_Customer')
    try:
        df_processed['Dt_Customer_1'] = pd.to_datetime(df_processed[DATE_COL], format='mixed', errors='coerce')
        min_date = df_processed['Dt_Customer_1'].min()
        df_processed['Dates'] = (df_processed['Dt_Customer_1'] - min_date).dt.days
    except Exception as e:
        print(f"Error processing date column {DATE_COL}: {e}")
        # Handle cases where date conversion might fail entirely - fill with 0 or median?
        df_processed['Dates'] = 0 # Simple fallback

    # Check for NaNs introduced by date conversion errors
    if df_processed['Dates'].isnull().any():
        print(f"Warning: NaNs found in 'Dates' column after conversion. Filling with 0.")
        df_processed['Dates'].fillna(0, inplace=True)


    # 2. Handle Categorical Features ('Education', 'Marital_Status')
    # Define mappings (ensure consistency with notebook)
    # Get unique values *after* potential NaN handling
    unique_edu = df_processed[EDU_COL].astype(str).fillna('Unknown').unique() # Treat NaN as 'Unknown' string
    unique_marital = df_processed[MARITAL_COL].astype(str).fillna('Unknown').unique()

    # Ensure 'Unknown' is captured if it resulted from fillna
    if 'Unknown' not in unique_edu:
        unique_edu = np.append(unique_edu, 'Unknown')
    if 'Unknown' not in unique_marital:
        unique_marital = np.append(unique_marital, 'Unknown')

    # Create fixed-size mappings based on constants N_EDU_FEATURES/N_MARITAL_FEATURES
    Education_map = {category: [1 if i == idx else 0 for i in range(N_EDU_FEATURES)]
                     for idx, category in enumerate(unique_edu) if idx < N_EDU_FEATURES}
    Marital_status_map = {category: [1 if i == idx else 0 for i in range(N_MARITAL_FEATURES)]
                          for idx, category in enumerate(unique_marital) if idx < N_MARITAL_FEATURES}

    # Add default mapping for any unexpected values or values exceeding the fixed size
    default_edu_encoding = [0] * N_EDU_FEATURES
    default_marital_encoding = [0] * N_MARITAL_FEATURES

    # Fill NaNs before applying map, then apply map with default for unknown/missing keys
    df_processed[EDU_COL] = df_processed[EDU_COL].astype(str).fillna('Unknown').apply(lambda x: Education_map.get(x, default_edu_encoding))
    df_processed[MARITAL_COL] = df_processed[MARITAL_COL].astype(str).fillna('Unknown').apply(lambda x: Marital_status_map.get(x, default_marital_encoding))

    # Keep only necessary columns (features + target)
    all_feature_cols = NUMERICAL_COLS + [EDU_COL, MARITAL_COL]
    if TARGET_COL in df_processed.columns:
        final_cols = all_feature_cols + [TARGET_COL]
    else:
        final_cols = all_feature_cols # For test data

    # Check if all expected columns exist after potential removals/errors
    missing_cols = [col for col in final_cols if col not in df_processed.columns and col not in [EDU_COL, MARITAL_COL]] # Check only non-transformed cols
    if missing_cols:
        print(f"Error: The following original columns are missing before final selection: {missing_cols}")
        raise ValueError(f"Missing columns needed for preprocessing: {missing_cols}")

    # Select final columns
    df_final = df_processed[final_cols].copy()

    # Note: Imputation is handled by the Imputation_layer within the model

    return df_final, id_values


# --- PyTorch Dataset ---

class CustomerDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # Extract features and target, convert to numpy for efficiency
        self.numerical_features = self.dataframe[NUMERICAL_COLS].values.astype(np.float32)
        # Stack the list of lists into a 2D numpy array
        self.edu_features = np.array(self.dataframe[EDU_COL].tolist(), dtype=np.float32)
        self.marital_features = np.array(self.dataframe[MARITAL_COL].tolist(), dtype=np.float32)

        if TARGET_COL in self.dataframe.columns:
            self.targets = self.dataframe[TARGET_COL].values.astype(np.float32)
        else:
            self.targets = None # No targets in test set

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        num = torch.tensor(self.numerical_features[idx, :], dtype=torch.float32)
        edu = torch.tensor(self.edu_features[idx, :], dtype=torch.float32)
        mar = torch.tensor(self.marital_features[idx, :], dtype=torch.float32)

        features = (num, edu, mar) # Group features for the model

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32) # Keep as scalar for BCEWithLogitsLoss
            return features, target
        else:
            return features # Return only features tuple for test set

# --- Save/Load Model Functions ---

def save_checkpoint(model, optimizer, epoch, val_accuracy, filename):
    """Save model checkpoint."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        # Store model config if needed for reloading, e.g. using ConfigurableModel
        # 'model_config': model.config # Assuming model stores its config
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint."""
    if not os.path.exists(filename):
        print(f"Checkpoint file {filename} does not exist")
        return None, 0

    try:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage) # Load to CPU first
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                 print(f"Warning: Could not load optimizer state dict, possibly due to model/param changes: {e}")
                 print("Optimizer state will be reset.")

        loaded_epoch = checkpoint.get('epoch', 0) # Default to 0 if not found
        loaded_val_acc = checkpoint.get('val_accuracy', 0.0) # Default to 0.0

        print(f"Checkpoint loaded: {filename} (epoch {loaded_epoch}, val_accuracy: {loaded_val_acc:.2f}%)")
        return loaded_epoch, loaded_val_acc
    except Exception as e:
        print(f"Error loading checkpoint from {filename}: {e}")
        return None, 0


# --- Training Loop (Removed manual L2 regularization) ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_dir=MODEL_DIR, resume_from=None):
    model.to(device)
    start_epoch = 0
    best_val_accuracy = 0.0
    best_model_path = os.path.join(save_dir, 'best_model_1.pth')

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Resume training from checkpoint if specified
    if resume_from is not None and os.path.exists(resume_from):
        loaded_epoch, loaded_best_acc = load_checkpoint(model, optimizer, resume_from)
        if loaded_epoch is not None:
            start_epoch = loaded_epoch # Resume from the epoch *after* the saved one
            best_val_accuracy = loaded_best_acc
            print(f"Resuming training from epoch {start_epoch + 1} with best validation accuracy: {best_val_accuracy:.2f}%")
        else:
             print(f"Could not load checkpoint {resume_from}, starting training from scratch.")


    print(f"Training started on device: {device}")

    # Use OneCycleLR learning rate scheduler for super-convergence
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'], # Use the optimizer's current LR as max_lr
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=25,   # Initial lr is max_lr/25
        final_div_factor=1000,  # Final lr is max_lr/1000
    )

    # Early stopping with increased patience
    early_stopping_patience = 100
    no_improvement_epochs = 0

    # Track metrics for monitoring
    val_losses = []
    val_accuracies = []

    # Enable automatic mixed precision training for faster training if available
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' and torch.cuda.is_available() else None
    if scaler:
        print("Using Automatic Mixed Precision (AMP).")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # --- Training Phase ---
        for batch_idx, batch_data in enumerate(train_loader):
            # Data is expected as (features_tuple, targets)
            inputs, targets = batch_data
            targets = targets.to(device)
            inputs = [x.to(device) for x in inputs] # Move feature tensors to device

            optimizer.zero_grad()

            # Use mixed precision training if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) # BCEWithLogitsLoss expects raw logits

                # Scale gradients and optimize
                scaler.scale(loss).backward()
                # Optional: Gradient Clipping (unscale first)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # Optional: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Update LR scheduler every batch (for OneCycleLR)
            scheduler.step()

            train_loss += loss.item()
            # For BCEWithLogitsLoss, apply sigmoid to get probabilities, then threshold
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                inputs, targets = batch_data
                targets = targets.to(device)
                inputs = [x.to(device) for x in inputs]

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Save metrics for monitoring
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}") # Print current learning rate
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Update best model if validation accuracy improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_epochs = 0
            # Save checkpoint epoch number as epoch + 1 (since epoch is 0-indexed)
            save_checkpoint(model, optimizer, epoch + 1, val_accuracy, best_model_path)
            print(f"  ** New best validation accuracy: {best_val_accuracy:.2f}% - Saved best model to {best_model_path} **")
        else:
            no_improvement_epochs += 1
            print(f"  ** No improvement. Current best validation accuracy: {best_val_accuracy:.2f}% ({no_improvement_epochs}/{early_stopping_patience} epochs) **")

        print("-" * 30)

        # Early stopping check
        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {no_improvement_epochs} epochs with no improvement.")
            break

    print(f"Training finished. Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model saved at: {best_model_path}")
    # Load the best model state before returning
    if os.path.exists(best_model_path):
         print("Loading best model weights...")
         load_checkpoint(model, None, best_model_path) # Load best weights into the model object
    else:
         print("Warning: Best model path not found after training.")

    return best_val_accuracy, best_model_path

# --- Prediction Function ---

def predict(model, test_loader, device):
    """Generate predictions using a trained model."""
    model.eval()  # Set model to evaluation mode
    model.to(device) # Ensure model is on the correct device
    all_predictions = []

    with torch.no_grad():  # No need to track gradients
        for batch_data in test_loader:
            # Data is expected as features_tuple
            inputs = batch_data
            inputs = [x.to(device) for x in inputs] # Move feature tensors to device

            # Forward pass
            outputs = model(inputs)

            # Apply sigmoid and threshold to get binary predictions
            predictions = (torch.sigmoid(outputs) >= 0.5).int().cpu().numpy()
            all_predictions.extend(predictions.tolist())

    return all_predictions

# --- Optuna Objective Function (using refactored model) ---

def objective(trial, train_data_df, val_data_df, device, epochs):
    """Objective function for Optuna hyperparameter search using ConfigurableModel."""

    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    dropout_small = trial.suggest_float("dropout_small", 0.05, 0.3)
    dropout_large = trial.suggest_float("dropout_large", 0.1, 0.5)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-7, 1e-4, log=True) # For optimizer weight_decay
    activation_name = trial.suggest_categorical("activation", ["SiLU", "ReLU", "GELU"])
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])

    # Layer configuration suggestion (pyramid architecture)
    n_layers = trial.suggest_int("n_layers", 3, 7) # Adjusted range slightly
    layers = []
    first_layer_size = trial.suggest_categorical("first_layer", [32, 64, 128])
    layers.append(first_layer_size)
    current_size = first_layer_size

    for i in range(1, n_layers):
        # Suggest increasing or same size for the first half, decreasing or same for the second
        ratio_options = [0.5, 1.0, 2.0] if i < n_layers / 2 else [0.5, 1.0]
        size_ratio = trial.suggest_categorical(f"layer_{i}_ratio", ratio_options)
        next_size = max(16, int(current_size * size_ratio)) # Ensure size doesn't drop too low
        # Ensure next_size is a power of 2 or reasonable value (optional refinement)
        # next_size = 2**int(np.log2(next_size)) # Force power of 2
        layers.append(next_size)
        current_size = next_size

    # Ensure the last layer size is reasonable (e.g., >= 16)
    if layers[-1] < 16:
        layers[-1] = 16

    print(f"\n--- Optuna Trial {trial.number} ---")
    print(f"  Params: lr={lr:.6f}, batch_size={batch_size}, dropout_small={dropout_small:.3f}, dropout_large={dropout_large:.3f}, weight_decay={l2_lambda:.2E}")
    print(f"  Network: Emb={embedding_dim}, Act={activation_name}, Layers={layers}")


    # 2. Create Datasets and DataLoaders
    try:
        train_dataset = CustomerDataset(train_data_df)
        val_dataset = CustomerDataset(val_data_df)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) # Added workers/pin_memory
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    except Exception as e:
        print(f"Error creating datasets/loaders for trial {trial.number}: {e}")
        # Pruning tells Optuna this trial failed and shouldn't be considered optimal
        raise optuna.exceptions.TrialPruned(f"Data loading failed: {e}")

    # 3. Initialize Model using ConfigurableModel
    try:
        model = ConfigurableModel(
            embedding_dim=embedding_dim,
            layers_list=layers,
            dropout_small=dropout_small,
            dropout_large=dropout_large,
            activation_name=activation_name
        )
    except Exception as e:
        print(f"Error initializing model for trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned(f"Model initialization failed: {e}")


    criterion = nn.BCEWithLogitsLoss()

    # 4. Choose Optimizer with weight_decay (L2 regularization)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RAdam", "Adam"]) # Prefer AdamW/RAdam for weight decay
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
    elif optimizer_name == "RAdam":
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    else: # Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda) # Adam also supports weight_decay


    # 5. Train the model
    try:
        # Create a trial-specific save directory for checkpoints
        trial_save_dir = os.path.join(MODEL_DIR, 'optuna_trials', f'trial_{trial.number}')
        # Train for the specified number of epochs for this trial
        validation_accuracy, _ = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_dir=trial_save_dir)

        # Optional: Optuna Pruning - stop trial early if it's performing poorly
        # trial.report(validation_accuracy, step=epochs) # Report final accuracy
        # if trial.should_prune():
        #      raise optuna.exceptions.TrialPruned()

    except optuna.exceptions.TrialPruned as e:
         print(f"Trial {trial.number} pruned.")
         raise e # Re-raise to signal Optuna
    except Exception as e:
        print(f"Error during training for trial {trial.number}: {e}")
        # Also prune if training crashes
        raise optuna.exceptions.TrialPruned(f"Training failed: {e}")

    # 6. Return the metric to be maximized
    return validation_accuracy


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Tune Customer Purchase Prediction Model')
    parser.add_argument('--mode', type=str, default='tune', choices=['train', 'tune', 'predict'],
                        help='Run mode: train with fixed/best params, tune with Optuna, or predict on test data')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs (per trial if tuning)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (used in train/predict mode, override if use_best_params)')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate (used in train/predict mode, override if use_best_params)')
    parser.add_argument('--val_split', type=float, default=DEFAULT_VAL_SPLIT, help='Fraction of data for validation')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=['cuda', 'cpu'], help='Device for training (cuda/cpu)')
    parser.add_argument('--data_path', type=str, default=TRAIN_CSV_PATH, help='Path to the training CSV file')
    parser.add_argument('--test_path', type=str, default=TEST_CSV_PATH, help='Path to the test CSV file (for predict mode)')
    parser.add_argument('--submission_path', type=str, default=SUBMISSION_CSV_PATH,
                       help='Path to save the submission CSV file (for predict mode)')
    parser.add_argument('--n_trials', type=int, default=DEFAULT_N_TRIALS, help='Number of Optuna trials (used in tune mode)')
    parser.add_argument('--use_best_params', action='store_true',
                       help='Use hardcoded best hyperparameters (BEST_LR, BEST_BATCH_SIZE) for train/predict modes')
    parser.add_argument('--model_path', type=str, default=BEST_MODEL_PATH,
                       help='Path to a saved model checkpoint to load (used in predict mode, or as default best model)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to a checkpoint to resume training from (for train mode)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    elif args.device == 'cuda':
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")


    print("--- Configuration ---")
    print(f"Mode: {args.mode}")

    # Set LR and Batch Size based on mode and flags
    current_lr = args.lr
    current_batch_size = args.batch_size
    if args.mode in ['train', 'predict'] and args.use_best_params:
        print(f"Using hardcoded best hyperparameters:")
        print(f"  Learning Rate: {BEST_LR}")
        print(f"  Batch Size: {BEST_BATCH_SIZE}")
        current_lr = BEST_LR
        current_batch_size = BEST_BATCH_SIZE
    elif args.mode in ['train', 'predict']:
         print(f"Using provided/default hyperparameters:")
         print(f"  Learning Rate: {current_lr}")
         print(f"  Batch Size: {current_batch_size}")


    print(f"Epochs: {args.epochs}")
    if args.mode == 'tune':
        print(f"Optuna Trials: {args.n_trials}")
    elif args.mode == 'predict':
        print(f"Test Data Path: {args.test_path}")
        print(f"Submission Path: {args.submission_path}")
        print(f"Model Path for Prediction: {args.model_path}")

    print(f"Validation Split: {args.val_split}")
    print(f"Device: {args.device}")
    print(f"Train Data Path: {args.data_path}")
    if args.resume_from and args.mode == 'train':
        print(f"Resuming training from checkpoint: {args.resume_from}")
    print("---------------------")


    # --- Prediction Mode ---
    if args.mode == 'predict':
        print("\n--- Starting Prediction Mode ---")

        # 1. Load & preprocess test data
        if not os.path.exists(args.test_path):
            print(f"Error: Test file not found at {args.test_path}")
            exit(1)
        try:
            test_df = pd.read_csv(args.test_path)
            print(f"Loaded test data: {test_df.shape}")
            processed_test_df, test_ids = preprocess_data(test_df)
            print(f"Preprocessed test data shape: {processed_test_df.shape}")
            if test_ids is None:
                print(f"Error: ID column ('{ID_COL}') not found or empty in test data.")
                exit(1)
        except Exception as e:
            print(f"Error loading/preprocessing test data: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

        # 2. Create test dataset and dataloader
        try:
            test_dataset = CustomerDataset(processed_test_df)
            # Use a reasonable batch size for prediction, can be larger than training
            predict_batch_size = current_batch_size * 2
            test_loader = DataLoader(test_dataset, batch_size=predict_batch_size, shuffle=False, num_workers=2, pin_memory=True)
            print(f"Test DataLoader created with batch size {predict_batch_size}.")
        except Exception as e:
            print(f"Error creating test dataloader: {e}")
            exit(1)

        # 3. Initialize model (using default config, state will be loaded)
        # If model saving included config, we could load dynamically here
        model = ConfigurableModel() # Uses default architecture params initially
        model.to(args.device)

        # 4. Load model from checkpoint
        model_load_path = args.model_path
        if not os.path.exists(model_load_path):
             print(f"Error: Model file not found at {model_load_path}")
             # Optional: Add logic here to train a model if none is found,
             # but for pure prediction mode, it's better to require a model.
             print("Please provide a valid path to a trained model using --model_path")
             # As a fallback, could try loading from MODEL_DIR/best_model.pth
             fallback_path = os.path.join(MODEL_DIR, 'best_model_1.pth')
             if os.path.exists(fallback_path):
                 print(f"Attempting to load fallback model: {fallback_path}")
                 model_load_path = fallback_path
             else:
                 exit(1)


        print(f"Loading model state from {model_load_path}")
        _, _ = load_checkpoint(model, None, model_load_path) # Optimizer state not needed for prediction

        # 5. Generate predictions
        print("\nGenerating predictions on test data...")
        predictions = predict(model, test_loader, args.device)

        # 6. Create submission file
        if len(test_ids) != len(predictions):
            print(f"Error: Number of test IDs ({len(test_ids)}) does not match number of predictions ({len(predictions)}).")
            exit(1)

        submission_df = pd.DataFrame({
            ID_COL: test_ids,
            TARGET_COL: predictions
        })

        # Save submission file
        try:
            submission_df.to_csv(args.submission_path, index=False)
            print(f"Submission file saved to {args.submission_path}")
        except Exception as e:
            print(f"Error saving submission file to {args.submission_path}: {e}")
            exit(1)

        # Display prediction statistics
        n_positive = sum(predictions)
        n_total = len(predictions)
        if n_total > 0:
            print(f"\nPrediction Statistics:")
            print(f"  Total Predictions: {n_total}")
            print(f"  Positive Predictions (1): {n_positive} ({n_positive/n_total*100:.2f}%)")
            print(f"  Negative Predictions (0): {n_total-n_positive} ({(n_total-n_positive)/n_total*100:.2f}%)")
        else:
            print("No predictions were generated.")

        print("\n--- Prediction Mode Finished ---")
        exit(0) # Exit successfully after prediction

    # --- Training & Tuning Modes ---

    # 1. Load Data
    if not os.path.exists(args.data_path):
        print(f"Error: Training file not found at {args.data_path}")
        exit(1)
    try:
        train_df_full = pd.read_csv(args.data_path)
        print(f"Loaded training data: {train_df_full.shape}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        exit(1)

    # 2. Preprocess Data
    print("Preprocessing data...")
    try:
        processed_df, _ = preprocess_data(train_df_full)
        print(f"Preprocessing complete. Processed data shape: {processed_df.shape}")
        # Sanity check for NaNs after preprocessing (should be handled by imputation layer)
        # if processed_df[NUMERICAL_COLS].isnull().values.any():
        #     print("Warning: NaNs found in numerical columns AFTER preprocessing. Imputation layer should handle.")
        # if processed_df[EDU_COL].isnull().any() or processed_df[MARITAL_COL].isnull().any():
        #     # This check might be too strict if the encoding creates lists/arrays
        #     pass
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
        exit(1)


    # 3. Split Data (Train/Validation)
    try:
        # Check if target column exists and has variance for stratification
        if TARGET_COL in processed_df.columns and processed_df[TARGET_COL].nunique() > 1:
             train_data, val_data = train_test_split(
                 processed_df,
                 test_size=args.val_split,
                 random_state=42,
                 stratify=processed_df[TARGET_COL]
             )
             print(f"Data split (stratified): Train {train_data.shape}, Validation {val_data.shape}")
        else:
             print("Warning: Cannot stratify split (target missing or has only one class). Performing random split.")
             train_data, val_data = train_test_split(
                 processed_df,
                 test_size=args.val_split,
                 random_state=42
             )
             print(f"Data split (random): Train {train_data.shape}, Validation {val_data.shape}")
    except Exception as e:
         print(f"Fatal error splitting data: {e}")
         exit(1)

    # --- Mode Selection: Train or Tune ---

    if args.mode == 'train':
        # 4a. Create Datasets and DataLoaders for Training
        print("\n--- Starting Training Mode ---")
        train_dataset = CustomerDataset(train_data)
        val_dataset = CustomerDataset(val_data)
        train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # 5a. Initialize Model, Loss, Optimizer
        # Using default architecture defined in ConfigurableModel unless loading a checkpoint overrides it
        model = ConfigurableModel()
        criterion = nn.BCEWithLogitsLoss()
        # Use AdamW with a default weight decay, can be adjusted
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-5) # Default weight decay

        # 6a. Train Model
        # The train_model function handles resuming if args.resume_from is set
        _, best_model_path = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            args.device, args.epochs, save_dir=MODEL_DIR, resume_from=args.resume_from
        )
        print(f"\nTraining finished. Best model saved to: {best_model_path}")

    elif args.mode == 'tune':
        # 4b. Set up Optuna Study
        print("\n--- Starting Tuning Mode with Optuna ---")
        # Use a persistent storage option like SQLite to save study progress
        study_name = "customer_prediction_study"
        storage_name = f"sqlite:///{study_name}.db"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True, # Resume study if it exists
            direction="maximize" # Maximize validation accuracy
        )
        print(f"Using Optuna study '{study_name}' stored at '{storage_name}'")


        # Use partial to pass fixed arguments to the objective function
        objective_with_data = partial(objective,
                                      train_data_df=train_data,
                                      val_data_df=val_data,
                                      device=args.device,
                                      epochs=args.epochs) # Use the same epochs per trial

        # 5b. Run Optuna Optimization
        try:
            study.optimize(objective_with_data, n_trials=args.n_trials, timeout=3600*4) # Added 4h timeout
        except KeyboardInterrupt:
             print("Optuna optimization stopped manually.")
        except optuna.exceptions.TrialPruned as e:
            print(f"A trial was pruned during optimization: {e}")
        except Exception as e:
            print(f"An error occurred during Optuna optimization: {e}")
            import traceback
            traceback.print_exc()

        # 6b. Print Results
        print("\n--- Optuna Tuning Finished ---")
        try:
            print(f"Number of finished trials: {len(study.trials)}")

            # Filter out pruned trials before finding the best one
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            if completed_trials:
                best_trial = study.best_trial # Note: study.best_trial considers only completed trials
                print(f"Best trial number: {best_trial.number}")
                print(f"Best trial value (Validation Accuracy): {best_trial.value:.4f}")
                print("Best hyperparameters:")
                for key, value in best_trial.params.items():
                    print(f"  {key}: {value}")

                # Save best parameters found (optional)
                best_params_path = os.path.join(MODEL_DIR, 'best_optuna_params.json')
                import json
                with open(best_params_path, 'w') as f:
                    json.dump(best_trial.params, f, indent=4)
                print(f"Best parameters saved to {best_params_path}")

                # Automatically train final model with best params found? Optional.
                # print("\nTraining final model with best parameters...")
                # best_params = best_trial.params
                # # Create loaders with best batch size
                # final_train_loader = DataLoader(CustomerDataset(train_data), batch_size=best_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
                # final_val_loader = DataLoader(CustomerDataset(val_data), batch_size=best_params['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
                # # Create model with best architecture params
                # final_model = ConfigurableModel(
                #      embedding_dim=best_params['embedding_dim'],
                #      layers_list=best_params['layers'], # Need to ensure layers are saved/reconstructed correctly
                #      dropout_small=best_params['dropout_small'],
                #      dropout_large=best_params['dropout_large'],
                #      activation_name=best_params['activation']
                # )
                # final_criterion = nn.BCEWithLogitsLoss()
                # # Create optimizer with best type, lr, weight_decay
                # optimizer_name = best_params['optimizer']
                # if optimizer_name == "AdamW":
                #     final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['l2_lambda'])
                # elif optimizer_name == "RAdam":
                #      final_optimizer = optim.RAdam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['l2_lambda'])
                # else:
                #      final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['l2_lambda'])

                # # Train the final model (potentially for more epochs)
                # _, final_best_model_path = train_model(final_model, final_train_loader, final_val_loader, final_criterion, final_optimizer, args.device, args.epochs * 2) # e.g., Train longer
                # print(f"Final best model from tuning saved to: {final_best_model_path}")
            else:
                 print("No trials completed successfully.")


        except ValueError:
            # This might happen if study.best_trial is accessed when no trials completed
            print("Optuna study finished without completing any successful trials.")
        except Exception as e:
            print(f"An error occurred retrieving Optuna results: {e}")
            import traceback
            traceback.print_exc()


    print("\n--- Script Finished ---") 