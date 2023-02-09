import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, count_of_moves):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            nn.ELU(),
            nn.Dropout(.2),

            nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            nn.ELU(),
            # nn.ReLU(),
            nn.Dropout(.2),

            nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.ELU(),
            nn.Dropout(.2),

            nn.Linear(hidden_size, count_of_moves)
        )
        

        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.network(x)
        return x
