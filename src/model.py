import torch
import torch.nn as nn


class CNNLSTMAttentionDecoder(nn.Module):
    def __init__(self, n_channels=64, segment_length=256, lstm_hidden_size=64, lstm_layers=1, num_classes=2,
                 dropout=0.3):
        """
        Args:
            n_channels (int): Number of EEG channels (e.g., 64)
            segment_length (int): Number of time samples per EEG segment (e.g., 256)
            lstm_hidden_size (int): Hidden size for LSTM layers
            lstm_layers (int): Number of stacked LSTM layers
            num_classes (int): Number of output classes (2 for binary classification)
            dropout (float): Dropout probability for regularization
        """
        super(CNNLSTMAttentionDecoder, self).__init__()

        # CNN part: 1D conv over time dimension for each channel
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # reduces time length by half

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # reduces time length by half again

        # Calculate the resulting sequence length after two pooling layers
        # Input length / 2 / 2 = input length / 4
        lstm_input_length = segment_length // 4

        # LSTM part: input size = number of CNN output channels; sequence length = lstm_input_length
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(lstm_hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, channels, samples) e.g. (B, 64, 256)
        """
        # CNN layers expect input as (batch, channels, time)
        x = self.conv1(x)  # -> (B, 32, segment_length)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # length reduced by 2

        x = self.conv2(x)  # -> (B, 64, segment_length/2)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)  # length reduced by 2 again -> total /4

        # Prepare for LSTM: transpose to (B, seq_len, features)
        x = x.permute(0, 2, 1)  # (B, length, channels), here channels=64 now features per step

        # LSTM expects (B, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, seq_len, hidden_size)

        # Use last output of LSTM sequence for classification
        out = lstm_out[:, -1, :]  # (B, hidden_size)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (B, num_classes)

        return out
