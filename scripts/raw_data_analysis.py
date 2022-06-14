import numpy as np
import matplotlib.pyplot as plt
import pickle
from creat_epoch_array import *


#  set parameters
lowcut = 1.
highcut = 40.

def load_data():
    #  get data from files
    with open("..\\recordings\\3\\trials.pickle", "rb") as f:
        raw_data = pickle.load(f)
    with open("..\\recordings\\3\\labels.pickle", "rb") as f:
        labels = pickle.load(f)
    with open("..\\recordings\\3\\targets.pickle", "rb") as f:
        targets = pickle.load(f)
    with open("..\\recordings\\3\\durations.pickle", "rb") as f:
        durations = pickle.load(f)

    return raw_data, labels, targets, durations


def split_data(data, ch_channels):
    # Append each
    trials = []
    for duration in durations:
        trial = []
        for start, end in duration:
            stim = data[ch_channels, start:end]
            trial.append(stim)
        trials.append(trial)
    return trials


raw_data, labels, targets, durations = load_data()

epochs = get_epochs_array(raw_data)
epochs.filter(l_freq=lowcut, h_freq=highcut, fir_design='firwin',
             skip_by_annotation='edge', verbose=False)
trials = split_data(epochs.get_data(), ch_channels)