import pandas as pd
from data_pipe import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model_def import SleepScoringLSTM
from model_def import params



def sleep_score_for_me(path, model='ssfm_v1', params=params):
    """Sleep score a numpy array consisting of the features needed for the specified model.

    Parameters
    ----------
    path : str
        path to the numpy file containing the features that you want to be scored.
    model : str, optional
        model to use, should be the name of a model in 'ssfm/models', by default 'ssfm_v1'
    """
    features = np.load(path)
    _features_tensor = torch.tensor(features, dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to the device (GPU or CPU)
    features_tensor = _features_tensor.to(device)
    model = SleepScoringLSTM(params[model]['input'], params[model]['hidden_dim'], params[model]['output_dim'], params[model]['num_layers'], params[model]['dropout_prob']).to(device)
    model.load_state_dict(torch.load(f'models/{model}.pth'))  # Replace 'model.pth' with your model file
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        predictions = model(features_tensor)
        _, predicted_labels = torch.max(predictions, 1)
    return predicted_labels