import torch

class MyRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self._hidden_size = hidden_size

        # Parameters used to compute the next hidden states
        self.wh = torch.nn.Linear(hidden_size, hidden_size)
        # Parameters used with the next input entry
        self.wx = torch.nn.Linear(input_size, hidden_size)
        # Parameters used with the computed hidden state to get the network output 
        self.wo = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initial hidden state will just be zeros
        hidden = torch.zeros(1, self._hidden_size)

        # For each entry in the input sequence
        for i in range(x.shape[0]):

            # Compute the new hidden state
            # h_1 = tanh((w_h * h_0) + (w_x * x_i))
            # where h_1 is the next hidden state, h_0 is the previous hidden state,
            # w_h and w_x are parameters and x_i is the i-th entry in the input sequence
            hidden = torch.tanh(self.wh(hidden) + self.wx(x[i].unsqueeze(0)))

            # Compute the network output for the i-th timestep
            output = self.wo(hidden)

        # Since this network will only output one value (0-1) after processing the entire input
        # sequence, we only compute the sigmoid of and return the final output, as opposed to one
        # output for each timestep. In other words, this is a many-to-one network
        return torch.sigmoid(output)
