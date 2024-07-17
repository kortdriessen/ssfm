import os
import numpy as np
import acr
import kdephys as kde
import tdt

from lspopt import spectrogram_lspopt
from functools import partial
get_spectrogram = partial(spectrogram_lspopt, c_parameter=20.)

#-----------------------------------------------------------------------------------------------------------
def robust_normalize(arr, p=1., axis=-1, method='standard score'):
    """
    References:
    -----------
    https://en.wikipedia.org/wiki/Normalization_(statistics)
    """

    # TODO: we might be able to do better than this...
    
    if method == 'min-max':
        return np.apply_along_axis(_robust_min_max_normalization, axis, arr, p=p)
    elif method == 'standard score':
        return np.apply_along_axis(_robust_standard_score_normalization, axis, arr, p=p)
    else:
        raise ValueError("Method one of: 'min-max', 'standard score'. Current value: {}".format(method))


def _robust_min_max_normalization(vec, p):
    minimum, maximum = np.percentile(vec, [p, 100-p])
    normalized = (vec - minimum) / (maximum - minimum)
    return normalized


def _robust_standard_score_normalization(vec, p):
    truncated = _truncate_signals(vec, p, 100.-p)
    robust_mean = np.mean(truncated)
    robust_std  = np.std(truncated)
    return (vec - robust_mean) / robust_std


def truncate_signals(arr, min_percentile=0.1, max_percentile=99.9, axis=0):
    return np.apply_along_axis(_truncate_signals, axis, arr,
                               min_percentile=min_percentile,
                               max_percentile=max_percentile)


def _truncate_signals(vec, min_percentile, max_percentile):
    min_value = np.percentile(vec, min_percentile)
    max_value = np.percentile(vec, max_percentile)
    vec[vec < min_value] = min_value
    vec[vec > max_value] = max_value
    return vec


#------------------------------------------------------------------------------------------------------------

def hypno_to_visbrain(hypno, file_path):
    def write_to_file(file_path, content):
        # Open the file in append mode ('a') to ensure we don't overwrite existing content
        with open(file_path, 'a') as file:
            # Write the formatted string and integer to the file
            file.write(content)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    write_to_file(file_path, f"*Duration_sec\t{hypno['duration'].sum()}\n")
    write_to_file(file_path, f"*Datafile\tUnspecified\n")
    for row in hypno.itertuples():
        write_to_file(file_path, f"{row.state}\t{row.end_time}\n")

def _resample_numpy(signal, desired_length):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, desired_length, endpoint=False),  # where to interpolate
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return(resampled_signal)

def get_sev_path(block_path, store, channel):
    for f in os.listdir(block_path):
        if f'{store}_Ch{channel}' in f:
            return os.path.join(block_path, f)

def v2uv(voltage_array):
    return voltage_array * 1e6


def prepare_raw_data(subject, recording, probe, channel, emg_store='EMGr', emg_chan=1, t1=0, t2_sub=1, target_fs=200):
    
    """Looads a single channel of lfp and emg data and resamples them to the target frequency.
    """
    
    si = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    rec_dur = si[recording]['duration']
    t2 = int(rec_dur - t2_sub) 
    block_path = acr.io.acr_path(subject, recording)
    sev_path = get_sev_path(block_path, probe, channel)
    lfp = tdt.read_sev(sev_path, t1=t1, t2=t2)
    lfp_dat = v2uv(lfp[probe].data)

    #ep = get_sev_path(block_path, emg_store, emg_chan)
    emg = tdt.read_block(block_path, t1=t1, t2=t2, store=emg_store, channel=emg_chan) # do this so as to not accidentally use the wrong sampling rate for emg
    emg_dat = v2uv(emg['streams'][emg_store].data)

    desired_length = (t2 - t1) * target_fs

    rslfp = _resample_numpy(lfp_dat, desired_length)
    rsemg = _resample_numpy(emg_dat, desired_length)
    
    return rslfp, rsemg


def spectrogram_from_raw_data(raw_dat, fs, time_res=1):
    """Computes the spectrogram of a raw data signal.
    """
    low_cut = 1
    high_cut = int((fs/2)-10)
    
    
    frequencies, time, spectrogram = get_spectrogram(raw_dat, 
                                                     fs=fs, 
                                                     noverlap=0,
                                                     nperseg=fs*time_res)
    
    mask = (frequencies >= low_cut) & (frequencies < high_cut)
    frequencies = frequencies[mask]
    spectrogram = spectrogram[mask]
    spectrogram = np.log(spectrogram + 1)
    spectrogram = robust_normalize(spectrogram, p=5., axis=0, method='standard score')
    return time, frequencies, spectrogram