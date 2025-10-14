"""
CNN Model for FlexTrack Challenge 2025
1D Convolutional Neural Network for time series prediction
"""

import torch
import torch.nn as nn
import numpy as np
from .base_dl_model import BaseDeepLearningModel


class CNNNetwork(nn.Module):
    """1D CNN Neural Network for Time Series"""

    def __init__(
        self,
        input_size: int,
        num_filters: list = [64, 128, 256],
        kernel_sizes: list = [3, 3, 3],
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """
        Initialize CNN Network

        Args:
            input_size: Number of input features
            num_filters: List of filter sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout: Dropout rate
            output_size: Number of output units
        """
        super(CNNNetwork, self).__init__()

        self.input_size = input_size

        # Build convolutional layers
        conv_layers = []
        in_channels = input_size

        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        fc_input_size = num_filters[-1] * 2  # *2 because we concat avg and max pooling

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

        # Convolutional layers
        x = self.conv_layers(x)

        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)

        # Concatenate pooled features
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Fully connected layers
        output = self.fc_layers(x)

        return output


class CNNModel(BaseDeepLearningModel):
    """CNN Model for Energy Flexibility Prediction"""

    def __init__(
        self,
        task: str = "classification",
        sequence_length: int = 96,
        num_filters: list = None,
        kernel_sizes: list = None,
        dropout: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = None,
    ):
        """
        Initialize CNN Model

        Args:
            task: 'classification' or 'regression'
            sequence_length: Number of time steps in input sequence
            num_filters: List of filter sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
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

        self.num_filters = num_filters if num_filters else [64, 128, 256]
        self.kernel_sizes = kernel_sizes if kernel_sizes else [3, 3, 3]
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
        Train CNN model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        # Initialize model with input size
        if self.model is None:
            self.input_size = X_train.shape[1]
            self.model = CNNNetwork(
                input_size=self.input_size,
                num_filters=self.num_filters,
                kernel_sizes=self.kernel_sizes,
                dropout=self.dropout,
                output_size=1,
            ).to(self.device)

            print(f"CNN Model Architecture:")
            print(f"  Input size: {self.input_size}")
            print(f"  Filters: {self.num_filters}")
            print(f"  Kernel sizes: {self.kernel_sizes}")
            print(
                f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

        # Call parent train method
        super().train(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    # Test CNN model
    print("Testing CNN Model...")

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
    clf = CNNModel(task="classification", epochs=5, sequence_length=sequence_length)
    clf.train(X_train, y_train, X_val, y_val)
    predictions = clf.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")

    # Regression
    print("\n=== Regression ===")
    y_train_reg = np.random.randn(n_samples) * 10
    y_val_reg = np.random.randn(200) * 10

    reg = CNNModel(task="regression", epochs=5, sequence_length=sequence_length)
    reg.train(X_train, y_train_reg, X_val, y_val_reg)
    predictions = reg.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    print("\nCNN Model test completed!")
