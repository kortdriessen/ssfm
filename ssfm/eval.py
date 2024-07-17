import pandas as pd
from ssfm.training_data_pipe import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from ssfm.model_def import params, SleepScoringLSTM

state_map = {
    0: 'NREM', 
    1: 'REM', 
    2: 'Wake',  
    3: 'Transition-to-REM', 
    4: 'Transition-to-NREM', 
    5: 'Transition-to-Wake', 
    6: 'Brief-Arousal', 
    7: 'Unsure',}

def predict_states(path, model_name='ssfm_v1', params=params):
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
    model = SleepScoringLSTM(params[model_name]['input_dim'], params[model_name]['hidden_dim'], params[model_name]['output_dim'], params[model_name]['num_layers'], params[model_name]['dropout_prob']).to(device)
    model.load_state_dict(torch.load(f'/home/kdriessen/gh_master/ssfm/ssfm/models/{model_name}.pth'))  # Replace 'model.pth' with your model file
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        predictions = model(features_tensor)
        _, predicted_labels = torch.max(predictions, 1)
    return predicted_labels.cpu().detach().numpy()

def get_change_indexes(df):
    """gets change positions

    Parameters
    ----------
    df : pd.DataFrame
        each row is an epoch-length (typically 2-second) prediction output from the model.
    """
    change_indexes = []
    for i, row in enumerate(df.itertuples()):
        if i == 0:
            running_state = row.state
        row_state = row.state
        if row_state == running_state:
            continue
        elif row_state != running_state:
            change_indexes.append(i)
            running_state = row_state
    change_indexes.append(df.index[-1])
    return change_indexes

def hypno_from_states_and_changes(df, change_indexes):
    hyp = pd.DataFrame(columns=['start_time', 'end_time', 'state', 'duration'])
    last = 0
    for i in change_indexes:
        print(i)
        if last == i:
            continue
        start_time = df.iloc[last:i]['start_time'].values[0]
        end_time = df.iloc[last:i]['end_time'].values[-1]
        duration = end_time - start_time
        state = df.iloc[last:i]['state'].values[0]
        hyp = pd.concat([hyp, pd.DataFrame({'start_time': start_time, 'end_time': end_time, 'state': state, 'duration': duration}, index=[0])], ignore_index=True)
        last = i
    return hyp

def hypno_to_text(hypno, file_path):
    def write_to_file(file_path, content):
        # Open the file in append mode ('a') to ensure we don't overwrite existing content
        with open(file_path, 'a') as file:
            # Write the formatted string and integer to the file
            file.write(content)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    write_to_file(file_path, f"*Duration_sec\t{hypno['duration'].sum()}\n")
    for row in hypno.itertuples():
        write_to_file(file_path, f"{row.state}\t{row.end_time}\n")

def predictions_to_hypno(predictions, state_map=state_map, epoch_length=2.0004864, save_path=None):
    states_array = [state_map[label.item()] for label in predictions]
    t = np.arange(0, len(states_array)*epoch_length, epoch_length)
    starts = t+epoch_length
    ends = starts+epoch_length
    df = pd.DataFrame({'start_time': starts, 'end_time': ends, 'state': states_array, 'duration': ends-starts})
    change_indexes = get_change_indexes(df)
    hyp = hypno_from_states_and_changes(df, change_indexes)
    hyp.loc[0, 'start_time'] = 0
    hyp.loc[len(hyp)-1, 'end_time'] = hyp.loc[len(hyp)-1, 'end_time']+epoch_length
    hyp['duration'] = hyp['end_time'] - hyp['start_time']
    if save_path:
        hypno_to_text(hyp, save_path)
    return hyp


def preds_to_hyp(states_array, times, epoch_length=1, save_path=None):
    starts = times-0.5
    ends = starts+epoch_length
    df = pd.DataFrame({'start_time': starts, 'end_time': ends, 'state': states_array, 'duration': ends-starts})
    change_indexes = get_change_indexes(df)
    hyp = hypno_from_states_and_changes(df, change_indexes)
    hyp.loc[0, 'start_time'] = 0
    hyp.loc[len(hyp)-1, 'end_time'] = hyp.loc[len(hyp)-1, 'end_time']+epoch_length
    hyp['duration'] = hyp['end_time'] - hyp['start_time']
    if save_path:
        hypno_to_text(hyp, save_path)
    return hyp

def sleep_score_for_me(path, model_name='ssfm_v1', params=params, save_path=None):
    predictions = predict_states(path, model_name=model_name, params=params)
    hyp = predictions_to_hypno(predictions, save_path=save_path)
    return hyp






def sleep_score_for_me(path, model_name='ssfm_v1', params=params, save_path=None):
    predictions = predict_states(path, model_name=model_name, params=params)
    hyp = predictions_to_hypno(predictions, save_path=save_path)
    return hyp

