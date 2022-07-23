import torch

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self._hidden_size = hidden_size

        self.rnn = torch.nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True    # To allow (batch_size, seq_length, input_size) shaped inputs
        )

        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
