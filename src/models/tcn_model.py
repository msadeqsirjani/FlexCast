"""
TCN Model for FlexTrack Challenge 2025
Temporal Convolutional Network for time series prediction
"""

import torch
import torch.nn as nn
import numpy as np
from .base_dl_model import BaseDeepLearningModel


class TemporalBlock(nn.Module):
    """Temporal Block for TCN"""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        """
        Initialize Temporal Block

        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Kernel size
            dilation: Dilation factor
            dropout: Dropout rate
        """
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation
        )
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # Residual connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNetwork(nn.Module):
    """Temporal Convolutional Network"""

    def __init__(
        self,
        input_size: int,
        num_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """
        Initialize TCN Network

        Args:
            input_size: Number of input features
            num_channels: List of channels for each TCN layer
            kernel_size: Kernel size
            dropout: Dropout rate
            output_size: Number of output units
        """
        super(TCNNetwork, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        fc_input_size = num_channels[-1] * 2  # *2 for avg and max pooling

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]

        Returns:
            Output tensor [batch_size, output_size]
        """
        # Reshape for Conv1d: [batch, channels, sequence]
        x = x.permute(0, 2, 1)

        # TCN layers
        x = self.network(x)

        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)

        # Concatenate pooled features
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Fully connected layers
        output = self.fc_layers(x)

        return output


class TCNModel(BaseDeepLearningModel):
    """TCN Model for Energy Flexibility Prediction"""

    def __init__(
        self,
        task: str = "classification",
        sequence_length: int = 96,
        num_channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = None,
    ):
        """
        Initialize TCN Model

        Args:
            task: 'classification' or 'regression'
            sequence_length: Number of time steps in input sequence
            num_channels: List of channels for each TCN layer
            kernel_size: Kernel size
            dropout: Dropout rate
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

        self.num_channels = num_channels if num_channels else [64, 128, 256]
        self.kernel_size = kernel_size
        self.dropout = dropout

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
        Train TCN model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        # Initialize model with input size
        if self.model is None:
            self.input_size = X_train.shape[1]
            self.model = TCNNetwork(
                input_size=self.input_size,
                num_channels=self.num_channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                output_size=1,
            ).to(self.device)

            print(f"TCN Model Architecture:")
            print(f"  Input size: {self.input_size}")
            print(f"  Channels: {self.num_channels}")
            print(f"  Kernel size: {self.kernel_size}")
            print(
                f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

        # Call parent train method
        super().train(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    # Test TCN model
    print("Testing TCN Model...")

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
    clf = TCNModel(task="classification", epochs=5, sequence_length=sequence_length)
    clf.train(X_train, y_train, X_val, y_val)
    predictions = clf.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")

    # Regression
    print("\n=== Regression ===")
    y_train_reg = np.random.randn(n_samples) * 10
    y_val_reg = np.random.randn(200) * 10

    reg = TCNModel(task="regression", epochs=5, sequence_length=sequence_length)
    reg.train(X_train, y_train_reg, X_val, y_val_reg)
    predictions = reg.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    print("\nTCN Model test completed!")
