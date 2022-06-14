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
