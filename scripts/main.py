from preprocess import preprocess_data
from down_sample import *
from plots import *

path = "..\\recordings\\Shiri_20.06\\\2"
trials, labels, targets = preprocess_data(path)

# plot before mean
plot_electrodes(trials[0], 4)

mean_target_data, mean_distract_data, mean_idle_data = get_mean_of_stims(trials, labels, targets)

down_sample_target = down_sample(mean_target_data, sample_rate)
down_sample_idle = down_sample(mean_idle_data, sample_rate)

# plot after mean
plot_electrodes(mean_target_data[0], 4)
plot_electrodes(mean_distract_data[0], 4)
plot_electrodes(mean_idle_data[0], 4)

# plot mean for all electrodes
# plot_mean_data(trials, mean_target_data, mean_distract_data, mean_idle_data)

# plot after down sample
plot_down_sample(trials, down_sample_target, down_sample_idle)

