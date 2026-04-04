import torch
import torch.nn as nn
from torch import Tensor


class AudioCNNFrontEnd(nn.Module):
    """
    Collapses a log-mel spectrogram [B, n_mels=512, T] into a sequence of
    d_model-dimensional vectors [B, T//16, d_model] via three strided Conv2d layers.

    Frequency reduction: 512 → 16 → 2 → 1  (fully collapsed)
    Time reduction:      T   → T/4 → T/8 → T/16
    """

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(32, 4), stride=(32, 4))
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(8, 2), stride=(8, 2))
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, d_model, kernel_size=(2, 2), stride=(2, 2))

        self.act = nn.GELU()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, n_mels, T]  (n_mels must be 512)
        Returns:
            [B, T//16, d_model]
        """
        # [B, n_mels, T] → [B, 1, n_mels, T]
        x = x.unsqueeze(1)

        x = self.act(self.bn1(self.conv1(x)))   # [B, 64,  16, T//4]
        x = self.act(self.bn2(self.conv2(x)))   # [B, 128,  2, T//8]
        x = self.conv3(x)                        # [B, d_model, 1, T//16]

        x = x.squeeze(2)                         # [B, d_model, T//16]
        x = x.permute(0, 2, 1)                  # [B, T//16, d_model]
        return self.norm(self.proj(x))
