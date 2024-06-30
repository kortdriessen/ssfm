import torch.nn as nn

params = {}
params['ssfm_v1'] = {}
params['ssfm_v1']['input_dim'] = 5
params['ssfm_v1']['hidden_dim'] = 128
params['ssfm_v1']['output_dim'] = 8
params['ssfm_v1']['num_layers'] = 2
params['ssfm_v1']['dropout_prob'] = 0.2
params['ssfm_v1']['learning_rate'] = 0.002
params['ssfm_v1']['num_epochs'] = 40
params['ssfm_v1']['batch_size'] = 128

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