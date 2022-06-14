from scipy.signal import lfilter, butter
import numpy as np
import matplotlib.pyplot as plt
import pickle
from creat_epoch_array import *
from bci4als_code.eeg import EEG


def get_mean_of_stims(data, labels, targets):
    mean_target_data, mean_distract_data, mean_idle_data = [], [], []
    for trial, label, target in zip(data, labels, targets):
        trial_data = trial
        if not isinstance(trial, np.ndarray):
            trial_data = trial.get_data()
        label_array = np.array(label)
        target_data = trial_data[label_array == target]
        idle_data = trial_data[label_array == 2]
        distract_data = trial_data[label_array == (1 - target)]
        mean_target_data.append(np.mean(target_data, axis=0))
        mean_idle_data.append(np.mean(idle_data, axis=0))
        mean_distract_data.append(np.mean(distract_data, axis=0))
    return mean_target_data, mean_distract_data, mean_idle_data


#  set parameters
ch_names = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O1', 'O2']
lowcut = 1.
highcut = 40.
notch = 50
fs = 125

#  get data from files
with open("..\\recordings\\Elad_02.06\\2\\trials.pickle", "rb") as f:
    trials = pickle.load(f)
with open("..\\recordings\\Elad_02.06\\2\\labels.pickle", "rb") as f:
    labels = pickle.load(f)
with open("..\\recordings\\Elad_02.06\\2\\targets.pickle", "rb") as f:
    targets = pickle.load(f)

# filter!
trials_filtered = []
for trial in trials:
    # t = np.array(trial,)
    raw = EEG._board_to_mne(trial, ch_names)
    # trials_array = mne.io.RawArray(trial)
    trials_filtered[trial] = EEG.filter_data(raw, notch, lowcut, highcut)


data = get_epochs_array(trials=trials)


def plot_electrodes(trial, elec):
    plt.plot(range(len(trial[elec])), trial[elec])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


mean_target_data, mean_distract_data, mean_idle_data = get_mean_of_stims(data, labels, targets)

plot_electrodes(mean_target_data[0], 3)
plot_electrodes(mean_idle_data[0], 3)
plot_electrodes(mean_distract_data[0], 3)


# trials_filtered.append(butter_bandpass_filter(trial, lowcut, highcut, 125, order=2))
# data = trials_filtered
elec_len = len(trials[0][0])
for i in range(elec_len):
    plt.figure()
    fig, axs = plt.subplots(3)
    fig.suptitle('electorde #' + str(i))
    # fig.set_size_inches(12, 8)
    axs[0].plot(range(len(mean_target_data[0][i])), mean_target_data[0][i])
    axs[0].set_title('Target')
    axs[1].plot(range(len(mean_idle_data[0][i])), mean_idle_data[0][i])
    axs[1].set_title('Idle')
    axs[2].plot(range(len(mean_distract_data[0][i])), mean_distract_data[0][i])
    axs[2].set_title('Distract')
    plt.xlabel("Time")
    for ax in axs.flat:
        ax.set(ylabel='Frequency')
    plt.show()

        # plot_electrodes(mean_target_data[0], i)
    # axs[1].plot_electrodes(mean_idle_data[0], i)
    # axs[2].plot_electrodes(mean_distract_data[0], i)




def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
