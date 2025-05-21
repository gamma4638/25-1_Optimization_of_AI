import torch
import torch.nn as nn
from typing import List

class LSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        lag: int,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_sizes[0],
            num_layers=len(hidden_sizes),
            dropout=dropout if len(hidden_sizes) > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[0], 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # out shape: (batch_size, sequence_length, hidden_size)
        
        # 마지막 시퀀스의 출력만 사용
        return self.fc(out[:, -1, :]).squeeze() 