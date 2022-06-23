import numpy as np

#  set parameters
sample_rate = 10


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
        sampled_electrode = []
        for electrode in trail:
            sampled_electrode.append(electrode[1::sample_rate])
        sampled_data.append(sampled_electrode)
    return sampled_data


# we give the classifier the down sampled results -  only the relevant electrodes
# the classifier should tell us if the patient was concentrated
# when he saw the target
