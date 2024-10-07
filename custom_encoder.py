import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory

class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.fc1 = nn.Linear(observation_shape[0], 400)
        self.lstm = nn.LSTM(400, 300, batch_first=True)
        self.fc2 = nn.Linear(300, feature_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = h.unsqueeze(1)  # Add a sequence dimension
        h, _ = self.lstm(h)
        h = h[:, -1, :]  # Use last output of LSTM
        return torch.relu(self.fc2(h))

    def get_feature_size(self):
        return self.feature_size


class CustomEncoderFactory(EncoderFactory):
    TYPE = 'custom'

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return CustomEncoder(observation_shape, self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}
