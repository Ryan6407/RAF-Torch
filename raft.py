import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class RaF(torch.nn.Module):
    """
    Implementation of the RAF activation function. See: https://arxiv.org/pdf/2208.14111.pdf
    """
    def __init__(self, m, n):
        super().__init__()
        self.m = m
        self.n = n

        self.m_weights = nn.Parameter(
                torch.tensor([1] * self.m, dtype=torch.float)
            )
        self.n_weights = nn.Parameter(
                torch.tensor([1] * self.n, dtype=torch.float)
            )
        
        torch.nn.init.normal_(self.m_weights)
        torch.nn.init.normal_(self.n_weights)

    def forward(self, inputs):
        for i in range(self.m):
            if i == 0:
                x = torch.pow(inputs, torch.tensor([i]).to(device)) * self.m_weights[0]
            else:
                x += torch.pow(inputs, torch.tensor([i]).to(device)) * self.m_weights[0]

        for i in range(self.n):
            if i == 0:
                x2 = torch.pow(inputs, torch.tensor([i]).to(device)) * self.n_weights[0]
            else:
                x2 += torch.pow(inputs, torch.tensor([i]).to(device)) * self.n_weights[0]

        x2 = 1 + torch.abs(x2)
        return x/x2
