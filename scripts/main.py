import numpy as np
from preprocess import preprocess_data
from down_sample import *
from plots import *
from classifier import csp_lda


def load_data(path, plot=False):
    trials, labels, targets = preprocess_data(path)

    mean_target_data, mean_distract_data, mean_idle_data = get_mean_of_stims(trials, labels, targets)

    down_sample_target = down_sample(mean_target_data, sample_rate)
    down_sample_idle = down_sample(mean_idle_data, sample_rate)

    # plot after mean
    # plot_electrodes(mean_target_data[0], 4)
    # plot_electrodes(mean_distract_data[0], 4)
    # plot_electrodes(mean_idle_data[0], 4)

    if plot:
        # plot mean for all electrodes
        plot_mean_data(trials, mean_target_data, mean_distract_data, mean_idle_data)

        # plot after down sample
        plot_down_sample(trials, down_sample_target, down_sample_idle)

    return down_sample_target, down_sample_idle

paths = [f"..\\recordings\\Tamar_22.06\\{i}" for i in range(1,5)]
targets, idles = [], []
for path in paths:
    down_sample_target, down_sample_idle = load_data(path, plot=False)
    targets.extend(down_sample_target)
    idles.extend(down_sample_idle)

# Prepare data for the classifier:
labels = np.concatenate([np.zeros(len(idles)), np.ones(len(targets))])
trials = np.vstack([np.array(idles), np.array(targets)])

csp_lda(trials, labels)