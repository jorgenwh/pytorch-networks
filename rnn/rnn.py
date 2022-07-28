import torch

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_length):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.rnn = torch.nn.RNN(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers,
                batch_first=True
        )
        self.wo = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        output = self.wo(output)
        return output, hidden


