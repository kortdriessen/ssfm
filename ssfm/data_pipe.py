import kdephys as kde
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import acr
import tdt
import os
import scipy.signal as signal
import xarray as xr
from sklearn.preprocessing import StandardScaler


def load_base_data(subject, rec, probe, channel):
    print(f'Loading data for {subject}--{rec}--{probe}-{channel}')
    path = acr.io.acr_path(subject, rec)
    h = acr.io.load_hypno(subject, rec, corrections=True, update=False, float=False)
    total_duration = (h.end_time.max() - h.start_time.min()).total_seconds()
    #assert total_duration > (3600*5), f'Recording too short, should be at least 5 hours {subject}--{rec}--{total_duration}'
    dat = kde.xr.io.get_data(path, store=probe, channel=channel)
    emg = kde.xr.io.get_data(path, store='EMGr', channel=1)
    
    dat = dat.sel(datetime=slice(h.start_time.min(), h.end_time.max()))
    emg = emg.sel(datetime=slice(h.start_time.min(), h.end_time.max()))
    
    ds_dat = kde.xr.utils.decimate(dat, q=24)
    
    espg = kde.xr.spectral.get_spextrogram(emg)
    emg_power = espg.sum(dim='frequency')
    return ds_dat, emg_power

def extract_features(ds_dat, emg_power):
    spg = kde.xr.spectral.get_spextrogram(ds_dat) 
    assert spg.shape[1] == emg_power.shape[0], f'spectral shapes do no match'
    delta_bp = kde.xr.spectral.get_bandpower(spg, (0.5, 3))
    theta_bp = kde.xr.spectral.get_bandpower(spg, (4, 7))
    sigma_bp = kde.xr.spectral.get_bandpower(spg, (11, 16))
    gamma_bp = kde.xr.spectral.get_bandpower(spg, (65, 100))
    bp_df = pd.DataFrame({'emg': emg_power.values, 'delta': delta_bp.values, 'theta': theta_bp.values, 'sigma': sigma_bp.values, 'gamma': gamma_bp.values, 'time': delta_bp.time.values})
    return bp_df


def norm_features(bp_df):
    # Select the columns to normalize
    columns_to_normalize = ['emg', 'delta', 'theta', 'sigma', 'gamma']

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the data
    bp_df[columns_to_normalize] = scaler.fit_transform(bp_df[columns_to_normalize])
    return bp_df

def get_state_labels(subject, rec, t):
    h = acr.io.load_hypno(subject, rec, corrections=True, update=False, float=True)
    states = kde.hypno.hypno.get_states(h, t)
    states = pd.DataFrame(states, columns=['state'])
    states.loc[states['state'] == 'NREM', 'state'] = 0
    states.loc[states['state'] == 'REM', 'state'] = 1
    states.loc[states['state'] == 'Wake', 'state'] = 2
    states.loc[states['state'] == 'Wake-Good', 'state'] = 2
    states.loc[states['state'] == 'Transition-to-REM', 'state'] = 3
    states.loc[states['state'] == 'Transition-to-NREM', 'state'] = 4
    states.loc[states['state'] == 'Transition-to-Wake', 'state'] = 5
    states.loc[states['state'] == 'Brief-Arousal', 'state'] = 6
    states.loc[states['state'].apply(lambda x: isinstance(x, str)), 'state'] = 7
    return states

def create_sequences(features, labels, seq_length=10, stride=1):
    X = []
    y = []
    
    padding = seq_length // 2
    features = np.pad(features, ((padding, padding), (0, 0)), mode='edge')
    labels = np.pad(labels, (padding, padding), mode='edge')
    
    for i in range(0, len(features) - seq_length + 1, stride):
        X.append(features[i:i + seq_length])
        y.append(labels[i + seq_length // 2])
    return np.array(X), np.array(y)

def full_pipe(subject, recording, probe, channel):
    ds_dat, emg_power = load_base_data(subject, recording, probe, channel)
    bp_df = extract_features(ds_dat, emg_power)
    bp_df = norm_features(bp_df)
    states = get_state_labels(subject, recording, bp_df['time'].values)
    features = bp_df[['delta', 'theta', 'sigma', 'gamma', 'emg']].values
    labels = states['state'].values
    xseq, yseq = create_sequences(features, labels, seq_length=20, stride=1)
    assert xseq.shape[0] == yseq.shape[0], f'tensor shapes do not match | {subject}--{recording}'
    np.save(f'training_data/features__{subject}--{recording}--{probe}{channel}.npy', xseq)
    np.save(f'training_data/labels__{subject}--{recording}--{probe}{channel}.npy', yseq)