import collections
import glob
import os

from matplotlib import pyplot as plt
from matplotlib import mlab
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import scipy.io
import scipy.stats

import activity_classifier_utils
from sklearn.ensemble import RandomForestClassifier

def LoadTroikaDataset():
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls


def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]


def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability. 

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estim0tes = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))

def Evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs, confs = [], []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
        # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return AggregateErrorMetric(errs, confs)

def RunPulseRateAlgorithm(data_fl, ref_fl):
    # Load data using LoadTroikaDataFile
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
    
    # Compute pulse rate estimates and estimation confidence.
    
    # Loading the reference file
    ground_truth = scipy.io.loadmat(ref_fl)['BPM0'].reshape(-1)
    
    # Sample Frequency
    fs = 125

    # Window to calculate reference pulse rate
    win_len = 8

    # Overlap between time windows
    win_shift = 2
    
    def bandpass_filter(signal):
        """
        Filter the signal between the minfreq and maxfreq
    
        Args: 
            signal: signal from PPG or Accelerometer
    
        Returns:
            ndarray
  
        """
        pass_band=(40 / 60.0, 240 / 60.0)
        b, a = scipy.signal.butter(3, pass_band, btype='bandpass', fs=fs)
        return scipy.signal.filtfilt(b, a, signal)

    # Bandpass ppg and acc signals 
    ppg = bandpass_filter(ppg)
    accx = bandpass_filter(accx)
    accy = bandpass_filter(accy)
    accz = bandpass_filter(accz)
    
    def cal_spectogram(signal):
        """
        Calculate PSD estimates
    
        Args:
            signal: PPG or Accelerometer signals
        Returns:
            psd: ndarray
            freqs:  ndarray
    
        """
        psd, freqs, t = mlab.specgram(signal, NFFT=8*fs, Fs=fs, noverlap=6*fs, pad_to=12*fs)
        psd = psd[(freqs >= 40 / 60.0) & (freqs <= 240 / 60.0)]
        freqs = freqs[(freqs >= 40 / 60.0) & (freqs <= 240 / 60.0)]
    
        return psd, freqs

    # Calculate Power Spectral Density estimates for each signal
    psd_ppg, freqs_ppg = cal_spectogram(signal=ppg)
    psd_accx, freqs_accx = cal_spectogram(signal=accx)
    psd_accy, freqs_accy = cal_spectogram(signal=accy)
    psd_accz, freqs_accz = cal_spectogram(signal=accz)
    
    # Get frequencies and time steps
    freqs = freqs_ppg.shape[0]
    time_steps = psd_ppg.shape[1]
    
    # Sorted index value for signals 
    ppg_index = (-psd_ppg).argsort(axis=0)
    accx_index = (-psd_accx).argsort(axis=0)
    accy_index = (-psd_accy).argsort(axis=0)
    accz_index = (-psd_accz).argsort(axis=0)
    
    estimated_freq = []
    for t in range(time_steps):
        for freq in range(freqs):
            i=0
            if freq == 2:
                estimated_freq.append(freqs_ppg[ppg_index[freq][t]])
                break
            elif np.all([(freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accx_index[i][t]]), 
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accy_index[i][t]]), 
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accz_index[i][t]]),
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accx_index[i+1][t]]),
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accy_index[i+1][t]]),
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accz_index[i+1][t]]),
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accx_index[i+2][t]]),
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accy_index[i+2][t]]),
                      (freqs_ppg[ppg_index[freq][t]] != freqs_ppg[accz_index[i+2][t]])]):
                estimated_freq.append(freqs_ppg[ppg_index[freq][t]])
                break
                
    
    def calc_snr(signal, hr_f):
        """
        Calculates the signal to noise ratio
    
        Args:
            signal: signal from PPG
            hr_f: heart rate frequency estimates
        
        Returns:
            int
      
        """
        n = len(signal)*2
        harmonic_f = hr_f * 2
        fft_mag = np.abs(np.fft.rfft(signal, n))
        freqs = np.fft.rfftfreq(n, 1/fs)
        window_f = 5/60
        fundamental_frequency_window = (freqs > hr_f - window_f) & (freqs < hr_f + window_f)
        harmonic_frequency_window = (freqs > harmonic_f - window_f) & (freqs < harmonic_f + window_f)
        signal = np.sum(fft_mag[(fundamental_frequency_window) | (harmonic_frequency_window)])
        noise = np.sum(fft_mag[~ ((fundamental_frequency_window) | (harmonic_frequency_window))])
        snr = signal / noise
    
        return snr

    # Claculate confidence
    confidence = []
    for i in range(len(estimated_freq)):
        snr = calc_snr(ppg, estimated_freq[i])
        confidence.append(snr)
        
    pre = np.array(estimated_freq) * 60
    
    # Claculate the absolute error 
    error_array = np.abs(ground_truth - pre)
    conf_array = np.array(confidence)
    
    # Return per-estimate absolute error and confidence as a 2-tuple of numpy arrays.
    return error_array, conf_array  
