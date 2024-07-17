import torch.nn as nn
import torch

params = {}
params['ssfm_v1'] = {}
params['ssfm_v1']['input_dim'] = 8 # the number of features in the input
params['ssfm_v1']['hidden_dim'] = 128 # the number of features in the hidden state
params['ssfm_v1']['output_dim'] = 6 # the number of features (sleep states) in the output
params['ssfm_v1']['num_layers'] = 1 # the number of LSTM layers
params['ssfm_v1']['dropout_prob'] = 0.0 # the dropout probability
params['ssfm_v1']['learning_rate'] = 0.001 # the learning rate
params['ssfm_v1']['num_epochs'] = 250 # the number of epochs to train the model
#params['ssfm_v1']['batch_size'] = 32 # the batch size

params['ssfm_v2'] = {}
params['ssfm_v2']['input_dim'] = 8 # the number of features in the input
params['ssfm_v2']['hidden_dim'] = 128 # the number of features in the hidden state
params['ssfm_v2']['output_dim'] = 6 # the number of features (sleep states) in the output
params['ssfm_v2']['num_layers'] = 2 # the number of LSTM layers
params['ssfm_v2']['dropout_prob'] = 0.1 # the dropout probability
#params['ssfm_v2']['learning_rate'] = 0.001 # the learning rate
params['ssfm_v2']['num_epochs'] = 150 # the number of epochs to train the model
params['ssfm_v2']['batch_size'] = 1024 # the batch size


params['ssfm_v3'] = {}
params['ssfm_v3']['input_dim'] = 8 # the number of features in the input
params['ssfm_v3']['hidden_dim'] = 256 # the number of features in the hidden state
params['ssfm_v3']['output_dim'] = 6 # the number of features (sleep states) in the output
params['ssfm_v3']['num_layers'] = 4 # the number of LSTM layers
params['ssfm_v3']['dropout_prob'] = 0.33 # the dropout probability
#params['ssfm_v3']['learning_rate'] = 0.001 # the learning rate
params['ssfm_v3']['num_epochs'] = 200 # the number of epochs to train the model
params['ssfm_v3']['batch_size'] = 1024 # the batch size

class SleepScoringLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.5):
        super(SleepScoringLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Use the output from the last time step
        return out