"""
Transformer Model for FlexTrack Challenge 2025
Transformer architecture for time series prediction
"""

import torch
import torch.nn as nn
import numpy as np
import math
from .base_dl_model import BaseDeepLearningModel


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize Positional Encoding

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, sequence_length, d_model]

        Returns:
            Encoded tensor
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerNetwork(nn.Module):
    """Transformer Neural Network"""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """
        Initialize Transformer Network

        Args:
            input_size: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            output_size: Number of output units
        """
        super(TransformerNetwork, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )

        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, 128),
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
        # Project input to d_model dimensions
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Take the last time step
        x = x[:, -1, :]

        # Output layers
        output = self.fc_layers(x)

        return output


class TransformerModel(BaseDeepLearningModel):
    """Transformer Model for Energy Flexibility Prediction"""

    def __init__(
        self,
        task: str = "classification",
        sequence_length: int = 96,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        device: str = None,
    ):
        """
        Initialize Transformer Model

        Args:
            task: 'classification' or 'regression'
            sequence_length: Number of time steps in input sequence
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
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

        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
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
        Train Transformer model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        # Initialize model with input size
        if self.model is None:
            self.input_size = X_train.shape[1]
            self.model = TransformerNetwork(
                input_size=self.input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                output_size=1,
            ).to(self.device)

            print(f"Transformer Model Architecture:")
            print(f"  Input size: {self.input_size}")
            print(f"  Model dimension: {self.d_model}")
            print(f"  Attention heads: {self.nhead}")
            print(f"  Encoder layers: {self.num_encoder_layers}")
            print(f"  Feedforward dim: {self.dim_feedforward}")
            print(
                f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

        # Call parent train method
        super().train(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    # Test Transformer model
    print("Testing Transformer Model...")

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
    clf = TransformerModel(
        task="classification", epochs=5, sequence_length=sequence_length
    )
    clf.train(X_train, y_train, X_val, y_val)
    predictions = clf.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")

    # Regression
    print("\n=== Regression ===")
    y_train_reg = np.random.randn(n_samples) * 10
    y_val_reg = np.random.randn(200) * 10

    reg = TransformerModel(task="regression", epochs=5, sequence_length=sequence_length)
    reg.train(X_train, y_train_reg, X_val, y_val_reg)
    predictions = reg.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    print("\nTransformer Model test completed!")
