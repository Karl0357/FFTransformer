import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting.metrics import QuantileLoss as PFQuantileLoss


class MLPLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=1, dropout=0., activation='relu', res_con=True):
        super(MLPLayer, self).__init__()
        self.kernel_size = kernel_size
        if self.kernel_size != 1:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size, padding=(kernel_size-1))
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size, padding=(kernel_size-1))
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.res_con = res_con

    def forward(self, x):
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        if self.kernel_size != 1:
            y = y[..., 0:-(self.kernel_size - 1)]
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        if self.kernel_size != 1:
            y = y[:, 0:-(self.kernel_size - 1), :]
        if self.res_con:
            return self.norm2(x + y)
        else:
            return y


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, :, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(-1))
        
        loss = torch.mean(torch.cat(losses, dim=-1))
        return loss

    def __repr__(self):
        return f"MultiQuantileLoss(quantiles={self.quantiles})"
    

def temporal_split(data, calib_size, val_size):
    total_size = len(data)
    max_start = total_size - (calib_size + val_size)
    start = np.random.randint(0, max_start)
    
    calib_data = data[start:start+calib_size]
    val_data = data[start+calib_size:start+calib_size+val_size]
    
    return calib_data, val_data