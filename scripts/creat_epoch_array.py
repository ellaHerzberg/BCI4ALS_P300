import mne
import numpy as np

ch_names = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O1', 'O2']

def get_epochs_array(trials):
    """
    :param trials: 4 dimensional array
    :return: epochs array
    """
    trials = reshape_trails(trials)
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
    epochs = []
    for trial in trials:
        # Mirror freqs
        # new_trials = np.concatenate((trial[:, :, ::-1], trial, trial[:, :, ::-1]), axis=2)
        epoch = mne.EpochsArray(trial, info)
        epoch.set_montage(mont1020_new)
        epochs.append(epoch)
    return epochs


def reshape_trails(trials):
    n_samples = np.inf
    new_trials = []
    for trial in trials:
        for stim in trial:
            n_samples = min(stim.shape[1], n_samples)
    for trial in trials:
        cut_stims = []
        for stim in trial:
            cut_stims.append(stim[:, :n_samples])
        new_trials.append(np.stack(cut_stims))
    return new_trials

