import torch
import deepxde as dde

# To enforce A > 0, we can wrap the FNN
class BloodFlowPINN(torch.nn.Module):
    def __init__(self, fnn):
        super().__init__()
        self.fnn = fnn
        self.softplus = torch.nn.Softplus()
        
    def forward(self, x):
        y = self.fnn(x)
        # Apply softplus to A_hat to keep it strictly positive, and shift slightly to avoid exactly 0
        A_hat = self.softplus(y[:, 0:1]) + 1e-4
        Q_hat = y[:, 1:2]
        return torch.cat([A_hat, Q_hat], dim=1)

