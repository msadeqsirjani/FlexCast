"""
Base Deep Learning Model for FlexTrack Challenge 2025
Provides common functionality for PyTorch-based models
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Force single-threaded execution to avoid segfaults
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class BaseDeepLearningModel:
    """Base class for PyTorch models"""

    def __init__(
        self,
        task: str = "classification",
        sequence_length: int = 96,  # 24 hours with 15-min intervals
        batch_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
    ):
        """
        Initialize base deep learning model

        Args:
            task: 'classification' or 'regression'
            sequence_length: Number of time steps in input sequence
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.task = task
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Set device with CUDA support
        if device is not None:
            self.device = torch.device(device)
        else:
            # Auto-detect: CUDA > CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("CUDA not available, using CPU")
        
        # Additional safety settings for macOS
        import platform
        if platform.system() == "Darwin" and self.device.type == "cpu":
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            # Disable MKLDNN which can cause segfaults on macOS
            try:
                torch.backends.mkldnn.enabled = False
            except:
                pass

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"PyTorch threads: {torch.get_num_threads()}")

        self.model = None
        self.optimizer = None
        self.scaler = StandardScaler()
        self.history = {"train_loss": [], "val_loss": []}

    def _get_loss_function(self):
        """Get appropriate loss function for task"""
        if self.task == "classification":
            return nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss()

    def _prepare_sequences(
        self, X: np.ndarray, y: np.ndarray, stride: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequential data for training using stride to reduce memory

        Args:
            X: Features [n_samples, n_features]
            y: Labels [n_samples]
            stride: Step size between sequences (None = 1, larger values reduce memory)

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if stride is None:
            # Use adaptive stride based on dataset size to manage memory
            total_samples = len(X)
            if total_samples > 20000:
                stride = 4  # Use every 4th sequence for large datasets
            elif total_samples > 10000:
                stride = 2  # Use every 2nd sequence for medium datasets
            else:
                stride = 1  # Use all sequences for small datasets
        
        n_features = X.shape[1]
        
        # Calculate number of sequences with stride
        indices = np.arange(0, len(X) - self.sequence_length + 1, stride)
        n_samples = len(indices)
        
        # Convert to float32 to reduce memory
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        # Build sequences one at a time to avoid large intermediate arrays
        X_list = []
        y_list = []
        
        batch_size = 500  # Process 500 sequences at a time
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Create sequences for this batch
            for idx in batch_indices:
                X_list.append(X[idx:idx+self.sequence_length])
                y_list.append(y[idx+self.sequence_length-1])
        
        X_sequences = np.stack(X_list, axis=0)
        y_sequences = np.array(y_list, dtype=np.float32)
        
        X_tensor = torch.from_numpy(X_sequences)
        y_tensor = torch.from_numpy(y_sequences)
        
        return (X_tensor, y_tensor)

    def _create_dataloader(
        self, X: torch.Tensor, y: torch.Tensor, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader"""
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Enable pin_memory for faster GPU transfer
        use_pin_memory = self.device.type == "cuda"
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,  # Keep 0 to avoid multiprocessing issues
            pin_memory=use_pin_memory,  # Enable for CUDA
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train_scaled, y_train)

        if X_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val_scaled, y_val)

        # Create dataloaders
        train_loader = self._create_dataloader(X_train_seq, y_train_seq, shuffle=True)

        if X_val is not None:
            val_loader = self._create_dataloader(X_val_seq, y_val_seq, shuffle=False)

        # Loss and optimizer
        criterion = self._get_loss_function()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\nTraining {self.__class__.__name__} for {self.task}...")
        print(f"Sequence length: {self.sequence_length}, Batch size: {self.batch_size}")
        print(f"Training samples: {len(X_train_seq)}, Features: {X_train.shape[1]}")
        print(f"Device: {self.device}")
        print("-" * 60)

        # Training loop with tqdm progress bars
        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Training phase
            self.model.train()
            train_loss = 0.0

            # Progress bar for batches
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", 
                            leave=False, unit="batch")
            
            for i, (batch_X, batch_y) in enumerate(batch_pbar):
                try:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # Forward pass
                    self.optimizer.zero_grad()

                    if i == 0 and epoch == 0:
                        # Test with a very small batch first on first epoch
                        with torch.no_grad():
                            test_out = self.model(batch_X[:1])

                    outputs = self.model(batch_X).squeeze(-1)

                except RuntimeError as e:
                    print(f"\n✗ RuntimeError during training: {e}")
                    print(f"  This may be a PyTorch/BLAS library issue on your system.")
                    raise

                # Calculate loss
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
                # Update batch progress bar
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation phase
            if X_val is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc="Validating", leave=False, unit="batch")
                    for batch_X, batch_y in val_pbar:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X).squeeze(-1)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                val_loss /= len(val_loader)
                self.history["val_loss"].append(val_loss)

                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'best_val': f'{best_val_loss:.4f}'
                })

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    tqdm.write(f"✓ New best model at epoch {epoch+1} - val_loss: {val_loss:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}'})
                
        epoch_pbar.close()
        
        print(f"\nTraining completed!")
        print(f"Final train loss: {self.history['train_loss'][-1]:.4f}")
        if X_val is not None:
            print(f"Final val loss: {self.history['val_loss'][-1]:.4f}")
            print(f"Best val loss: {best_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features to predict

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        self.model.eval()

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create dummy y for sequence preparation
        y_dummy = np.zeros(len(X))
        X_seq, _ = self._prepare_sequences(X_scaled, y_dummy)

        # Create dataloader
        dataloader = self._create_dataloader(
            X_seq, torch.zeros(len(X_seq)), shuffle=False
        )

        predictions = []

        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X).squeeze(-1)

                if self.task == "classification":
                    # Apply sigmoid for binary classification
                    outputs = torch.sigmoid(outputs)

                predictions.append(outputs.cpu().numpy())

        predictions = np.concatenate(predictions)

        # Pad predictions to match input length
        padded_predictions = np.zeros(len(X))
        padded_predictions[: self.sequence_length - 1] = predictions[0]
        padded_predictions[self.sequence_length - 1 :] = predictions

        if self.task == "classification":
            # Convert probabilities to binary predictions
            return (padded_predictions > 0.5).astype(int)
        else:
            return padded_predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for classification

        Args:
            X: Features to predict

        Returns:
            Probability predictions
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification tasks")

        self.model.eval()

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create dummy y for sequence preparation
        y_dummy = np.zeros(len(X))
        X_seq, _ = self._prepare_sequences(X_scaled, y_dummy)

        # Create dataloader
        dataloader = self._create_dataloader(
            X_seq, torch.zeros(len(X_seq)), shuffle=False
        )

        predictions = []

        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X).squeeze(-1)
                outputs = torch.sigmoid(outputs)
                predictions.append(outputs.cpu().numpy())

        predictions = np.concatenate(predictions)

        # Pad predictions to match input length
        padded_predictions = np.zeros(len(X))
        padded_predictions[: self.sequence_length - 1] = predictions[0]
        padded_predictions[self.sequence_length - 1 :] = predictions

        # Return as 2D array with probabilities for both classes
        return np.column_stack([1 - padded_predictions, padded_predictions])

    def save_model(self, filepath: str):
        """Save model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "history": self.history,
            "task": self.task,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint["scaler"]
        self.history = checkpoint["history"]
        self.task = checkpoint["task"]
        self.sequence_length = checkpoint["sequence_length"]
        self.batch_size = checkpoint["batch_size"]
        self.learning_rate = checkpoint["learning_rate"]

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {filepath}")
