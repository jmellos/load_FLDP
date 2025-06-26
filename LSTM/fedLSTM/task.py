import os
import torch
from torch.nn import LSTM
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F

class load_model(torch.nn.Module):
    def __init__(
        self,
        input_size: int,    # number of features
        units: int = 64,
        output_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
  
        self.lstm = LSTM(
            input_size=input_size,    # number of features
            hidden_size=units,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(units, units // 2)
        self.fc2 = torch.nn.Linear(units // 2, units // 4)
        self.fc3 = torch.nn.Linear(units // 4, output_size)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]         # (batch, units)
        h = self.dropout(last_hidden)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def get_weights(net):
    state_dict = net.state_dict()
    return [val.cpu().numpy() for val in state_dict.values()]


def set_weights(net, parameters):
    keys = list(net.state_dict().keys())
    state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in zip(keys, parameters)
    })
    net.load_state_dict(state_dict, strict=True)


def train(net, train_loader, optimizer, device, epochs):
    criterion = torch.nn.MSELoss()
    net.to(device).train()


    for _ in range(epochs):
        for X, y in tqdm(train_loader, desc="Training"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = net(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()



def test(net, test_loader, device):
    
    criterion = torch.nn.MSELoss()
    net.to(device).eval()

    total_loss = 0.0
    total_mae  = 0.0
    n_samples  = 0

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            preds = net(X)

            batch_size = X.size(0)
            total_loss += criterion(preds, y).item() * batch_size
            total_mae  += torch.abs(preds - y).sum().item()
            n_samples  += batch_size

    mse = total_loss / n_samples
    mae = total_mae / n_samples
    return mse, mae


