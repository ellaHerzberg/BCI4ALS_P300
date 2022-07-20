import numpy as np
import mne
import pickle

#  set parameters
lowcut = 1.
highcut = 30.
ch_channels = list(range(1, 14))
ch_names = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O1', 'O2']
before_stim = int(0.2*75)
sample_rate = 10

def load_data(path):
    #  get data from files
    with open(path + "\\raw_data.pickle", "rb") as f:
        raw_data = pickle.load(f)
    with open(path + "\\labels.pickle", "rb") as f:
        labels = pickle.load(f)
    with open(path + "\\targets.pickle", "rb") as f:
        targets = pickle.load(f)
    with open(path + "\\durations.pickle", "rb") as f:
        durations = pickle.load(f)
    with open(path + "\\trials.pickle", "rb") as f:
        trials = pickle.load(f)

    return raw_data, labels, targets, durations, trials


def split_data(data, ch_channels, durations):
    # Append each
    trials = []
    for duration in durations:
        trial = []
        for start, end in duration:
            stim = data[ch_channels, max(0,start-before_stim):end]
            trial.append(stim)
        trials.append(trial)
    return trials


def reshape_trails(trials):
    n_samples = np.inf
    new_trials = []
    # find min num of stims for each trail
    for trial in trials:
        for stim in trial:
            n_samples = min(stim.shape[1], n_samples)
    # reshape the trails
    for trial in trials:
        cut_stims = []
        for stim in trial:
            cut_stims.append(stim[:, :n_samples])
        new_trials.append(np.stack(cut_stims))
    return new_trials


def get_epochs_array(trials):
    """
    :param trials: 4 dimensional array
    :return: list of epoch array for each trial
    """
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=125, ch_types=ch_types)

    # set montage
    mont1020 = mne.channels.make_standard_montage('standard_1020')

    mont1020_new = mont1020.copy()
    # Keep only the desired channels
    ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in ch_names]
    mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x+3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    mont1020_new.dig = mont1020.dig[0:3]+kept_channel_info
    # mont1020_new.plot()
    # Mirror freqs
    # new_trials = np.concatenate((trial[:, :, ::-1], trial, trial[:, :, ::-1]), axis=2)
    epoch = mne.EpochsArray(trials, info)
    epoch.set_montage(mont1020_new)
    return epoch


def remove_baseline(trials):
    new_trials = []
    for trial in trials:
        new_trial=[]
        for stim in trial:
            new_stim = []
            for electrode in stim:
                baseline = np.mean(electrode[0:before_stim])
                new_stim.append(electrode-baseline)
            new_trial.append(np.array(new_stim))
        new_trials.append(new_trial)
    return new_trials


def preprocess_data(path):
    raw_data, labels, targets, durations, trials = load_data(path)
    return general_preprocess(raw_data, durations, labels, targets)


def general_preprocess(data, durations, labels, targets=None):
    raw_data = data[np.newaxis, ...]
    epochs = get_epochs_array(raw_data[:, 1:14, :])
    epochs.filter(l_freq=lowcut, h_freq=highcut, verbose=False)

    raw_data[:, 1:14, :] = epochs.get_data()

    trials = split_data(raw_data.squeeze(0), ch_channels, durations)
    baseline_trials = remove_baseline(trials)
    trials = reshape_trails(baseline_trials)

    return trials, labels, targets


def get_mean_of_stims(data, labels, targets):
    """
    Calculate mean for every type of stimulus
    """
    mean_target_data, mean_distract_data, mean_idle_data = [], [], []

    for trial, label, target in zip(data, labels, targets):
        trial_data = trial
        if not isinstance(trial, np.ndarray):
            trial_data = trial.get_data()

        label_array = np.array(label)

        # split for different type of stimulus
        target_data = trial_data[label_array == target]
        idle_data = trial_data[label_array == 2]
        distract_data = trial_data[label_array == (1 - target)]

        # calc mean for every trial
        mean_target_data.append(np.mean(target_data, axis=0))
        mean_idle_data.append(np.mean(idle_data, axis=0))
        mean_distract_data.append(np.mean(distract_data, axis=0))

    return mean_target_data, mean_distract_data, mean_idle_data


def down_sample(mean_data, sample_rate):
    """
    Extract every n'th (=sample_rate) sample
    """
    sampled_data = []
    for trail in mean_data:
        sampled_trial = []
        for electrode in trail:
            sampled_trial.append(electrode[1::sample_rate])
        sampled_data.append(sampled_trial[0:8]) # take only the 8 first electrodes
    return sampled_data
