from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hlayers = 4, dropout = 0.1):
        super().__init__()

        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]

        for _ in range(num_hlayers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)