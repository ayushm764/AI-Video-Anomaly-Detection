import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """
    CNN encoder to extract spatial features from each frame.
    Uses a pretrained ResNet18 backbone.
    """

    def __init__(self, feature_dim=512):
        super().__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_dim = feature_dim

        # Freeze CNN weights (important for small datasets)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            feature vector of shape (B, feature_dim)
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for temporal anomaly detection.
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=1):
        super().__init__()

        # LSTM Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # LSTM Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=feature_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, feature_dim)

        Returns:
            Reconstructed features of shape (B, T, feature_dim)
        """
        # Encode temporal sequence
        encoded_seq, (hidden, cell) = self.encoder_lstm(x)

        # Repeat the final hidden state for each timestep
        repeated_hidden = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)

        # Decode the sequence
        reconstructed_seq, _ = self.decoder_lstm(repeated_hidden)

        return reconstructed_seq


class VideoAnomalyModel(nn.Module):
    """
    Full model combining CNN Encoder and LSTM Autoencoder.
    """

    def __init__(self):
        super().__init__()

        self.cnn_encoder = CNNEncoder()
        self.lstm_autoencoder = LSTMAutoencoder()

    def forward(self, clips):
        """
        Args:
            clips: Tensor of shape (B, T, C, H, W)

        Returns:
            Reconstructed feature sequences
        """
        B, T, C, H, W = clips.shape

        # Extract spatial features frame by frame
        features = []
        for t in range(T):
            frame = clips[:, t, :, :, :]
            feat = self.cnn_encoder(frame)
            features.append(feat)

        # Stack features across time
        features = torch.stack(features, dim=1)

        # Temporal reconstruction
        reconstructed = self.lstm_autoencoder(features)

        return features, reconstructed
