"""
GRU Model for FlexTrack Challenge 2025
Gated Recurrent Unit neural network for time series prediction
"""

import torch
import torch.nn as nn
import numpy as np
from .base_dl_model import BaseDeepLearningModel


class GRUNetwork(nn.Module):
    """GRU Neural Network"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,  # Disabled to prevent segfaults on macOS
        output_size: int = 1,
    ):
        """
        Initialize GRU Network

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Use bidirectional GRU
            output_size: Number of output units
        """
        super(GRUNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected layers
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.fc_layers = nn.Sequential(
            nn.Linear(gru_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]

        Returns:
            Output tensor [batch_size, output_size]
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)

        # Take the last time step output
        last_output = gru_out[:, -1, :]

        # Fully connected layers
        output = self.fc_layers(last_output)

        return output


class GRUModel(BaseDeepLearningModel):
    """GRU Model for Energy Flexibility Prediction"""

    def __init__(
        self,
        task: str = "classification",
        sequence_length: int = 96,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,  # Disabled to prevent segfaults on macOS
        batch_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = None,
    ):
        """
        Initialize GRU Model

        Args:
            task: 'classification' or 'regression'
            sequence_length: Number of time steps in input sequence
            hidden_size: Number of hidden units
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Use bidirectional GRU
            batch_size: Training batch size
            learning_rate: Learning rate
            epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            device: Device to use
        """
        super().__init__(
            task=task,
            sequence_length=sequence_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            device=device,
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.input_size = None
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """
        Train GRU model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        # Initialize model with input size
        if self.model is None:
            self.input_size = X_train.shape[1]
            self.model = GRUNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                output_size=1,
            ).to(self.device)

            print(f"GRU Model Architecture:")
            print(f"  Input size: {self.input_size}")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.num_layers}")
            print(f"  Bidirectional: {self.bidirectional}")
            print(
                f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

        # Call parent train method
        super().train(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    # Test GRU model
    print("Testing GRU Model...")

    # Create dummy data
    n_samples = 1000
    n_features = 20
    sequence_length = 96

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)

    X_val = np.random.randn(200, n_features)
    y_val = np.random.randint(0, 2, 200)

    # Classification
    print("\n=== Classification ===")
    clf = GRUModel(task="classification", epochs=5, sequence_length=sequence_length)
    clf.train(X_train, y_train, X_val, y_val)
    predictions = clf.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")

    # Regression
    print("\n=== Regression ===")
    y_train_reg = np.random.randn(n_samples) * 10
    y_val_reg = np.random.randn(200) * 10

    reg = GRUModel(task="regression", epochs=5, sequence_length=sequence_length)
    reg.train(X_train, y_train_reg, X_val, y_val_reg)
    predictions = reg.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    print("\nGRU Model test completed!")
