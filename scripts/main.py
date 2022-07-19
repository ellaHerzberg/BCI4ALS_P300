from preprocess import *
from plots import *
from classifier import csp_lda


def load_data(path, plot=False):
    trials, labels, targets = preprocess_data(path)

    mean_target_data, mean_distract_data, mean_idle_data = get_mean_of_stims(trials, labels, targets)

    down_sample_target = down_sample(mean_target_data, sample_rate)
    down_sample_idle = down_sample(mean_idle_data, sample_rate)

    if plot:
        # plot mean for all electrodes
        plot_all_mean_data(mean_target_data, mean_distract_data, mean_idle_data)

        # plot after down sample
        plot_all_down_sample(down_sample_target, down_sample_idle)

    return down_sample_target, down_sample_idle


if __name__=='__main__':
    paths = [f"..\\recordings\\Tamar_27.06\\{i}" for i in range(1, 4)]
    targets, idles = [], []
    for path in paths:
        down_sample_target, down_sample_idle = load_data(path, plot=False)
        targets.extend(down_sample_target)
        idles.extend(down_sample_idle)

    # Prepare data for the classifier:
    labels = np.concatenate([np.zeros(len(idles)), np.ones(len(targets))])
    trials = np.vstack([np.array(idles), np.array(targets)])

    csp_lda(trials, labels)

