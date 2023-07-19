# Data analysis
import numpy as np

import torch.nn as nn
from torch.utils.data import Dataset


class Normalizer:
    """
    Normalize data to standardize the range of the values before feeding them into the model.
    """

    def __init__(self):
        self.mu = None
        self.sd = None

    # Transformers
    def fit_transform(self, x):
        """
        Transforming all the features using the mean and variance of the training data.
        """
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        """
        Scale back the data to its original representation.
        """
        inverse_transform_x = x * self.sd + self.mu
        return inverse_transform_x


class LSTMModel(nn.Module):
    """
    Learn more about the model here: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    # Initialize the layers
    def __init__(
        self,
        input_size=1,
        hidden_layer_size=32,
        num_layers=2,
        output_size=1,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTM(
            hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for the linear layers.
        """
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        batchsize = x.shape[0]

        # Layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Reshape from hidden cell state
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # Layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


class TimeSeriesDataset(Dataset):
    """
    Data loader
    """

    # constructors
    def __init__(self, x, y):
        x = np.expand_dims(
            x, 2
        )  # Convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    # length
    def __len__(self):
        return len(self.x)

    # getter
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
