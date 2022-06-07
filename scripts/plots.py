from scipy.signal import lfilter, butter
import numpy as np
import matplotlib.pyplot as plt
import pickle
from creat_epoch_array import *


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
lowcut = 1.
highcut = 40.

#  get data from files
with open("..\\recordings\\Elad_02.06\\2\\trials.pickle", "rb") as f:
    trials = pickle.load(f)
with open("..\\recordings\\Elad_02.06\\2\\labels.pickle", "rb") as f:
    labels = pickle.load(f)
with open("..\\recordings\\Elad_02.06\\2\\targets.pickle", "rb") as f:
    targets = pickle.load(f)


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


#  filter function from mne
data = get_epochs_array(trials=trials)


def plot_electrodes(trial, elec):
    plt.plot(range(len(trial[elec])), trial[elec])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


mean_target_data, mean_distract_data, mean_idle_data = get_mean_of_stims(data, labels, targets)

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

# trials_filtered = []
# for trial in data:
#     trial.filter(l_freq=lowcut, h_freq=highcut, fir_design='firwin',
#                  skip_by_annotation='edge', verbose=False)
# trials_filtered.append(butter_bandpass_filter(trial, lowcut, highcut, 125, order=2))
# data = trials_filtered


# fs = 125  # (Hz)
# elec = len(trials[0][0])
# trials_amount = len(trials)
# stimuli = len(trials[1])
# labels_amount = max(labels[0])+1
# trials = np.random.randint(0, 100, (trials_amount, elec, fs * stimuli))
# labels = np.random.randint(low=0, high=labels_amount, size=(trials_amount, stimuli))
# trials = np.reshape(trials, (trials_amount, elec, fs, stimuli))

# times = np.arange(256)
# filtered = butter_bandpass_filter(trials, lowcut, highcut, fs, order=2)
# plt.subplot(121)
# plt.plot(times, trials[1, 1, 1, :])
# plt.xlabel('before')
# plt.subplot(122)
# # plt.plot(times, filtered[1, 1, :, 1])
# plt.xlabel('after')
# plt.show()

# making the length of the trails constant - reshape_trials do that

# mean_target_data, mean_distract_data, mean_idle_data = get_mean_of_stims(data, labels, targets)

# plot_electrodes(mean_target_data[0], 3)
# plot_electrodes(mean_idle_data[0], 3)
# plot_electrodes(mean_distract_data[0], 3)

# divide the data per trail per stimuli and calculate mean per stimuli
mean_data = []
# for t in range(trials_amount):
#     data_set = trials[t]
#     labels_set = np.array((labels[t]))
#     mean_trials = []
#     for j in range(labels_amount):
#         labels_trails = []
#         for k in range(stimuli):
#             if labels_set[k] == j:
#                 labels_trails.append(data_set[k])
#         mean_stimuli = np.zeros((len(labels_trails), 13, 60))
#         for d in range(len(labels_trails)):
#             mean_stimuli[d, :, :] = labels_trails[d]
#         mean_trials.append(np.mean(mean_stimuli, 0))
#     mean_data.append(mean_trials)
