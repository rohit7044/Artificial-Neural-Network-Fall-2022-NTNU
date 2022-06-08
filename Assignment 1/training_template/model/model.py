import torch.nn as nn
import torch


class LSTMPredictor(nn.Module):

    def __init__(self, look_back, num_layers=2, dropout=0.5, bidirectional=True):
        super(LSTMPredictor, self).__init__()

        # Nerual Layers
        self.rnn   = nn.LSTM(look_back, 32, num_layers, dropout=dropout, bidirectional=True)
        self.ly_a  = nn.Linear(32*(2 if bidirectional else 1), 16)
        # self.ly_a  = nn.Linear(look_back, 16)
        self.relu  = nn.ReLU()
        self.reg   = nn.Linear(16, 1)

    def predict(self, input):
        with torch.no_grad():
            return self.forward(input).item()

    def forward(self, input):
        r_out, (h_n, h_c) = self.rnn(input.unsqueeze(1), None)
        # print(r_out.shape)
        # input()
        logits = self.reg(self.relu(self.ly_a(r_out.squeeze(1))))
        # logits = self.reg(self.relu(self.ly_a(input)))

        return logits