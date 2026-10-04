"""PyTorch RUL regressors. Input (batch, window, features) -> output (batch,) in cycles."""
import torch
from torch import nn

from pmaint.features.rul import DEFAULT_RUL_CAP


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2, scale=DEFAULT_RUL_CAP):
        super().__init__()
        self.scale = scale
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1) * self.scale


class CNN1DRegressor(nn.Module):
    def __init__(self, n_features, channels=64, kernel=5, dropout=0.2, scale=DEFAULT_RUL_CAP):
        super().__init__()
        self.scale = scale
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=pad), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad), nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad), nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels * 2, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))  # (B, C, T)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # avg + max pooling over time
        return self.head(h).squeeze(-1) * self.scale


MODELS = {"lstm": LSTMRegressor, "cnn": CNN1DRegressor}


def build_model(name: str, n_features: int, **kw) -> nn.Module:
    return MODELS[name](n_features, **kw)
