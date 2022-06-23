import matplotlib.pyplot as plt


def plot_electrodes(trial, elec):
    """
    plot single electrode from single trial
    """
    plt.plot(range(len(trial[elec])), trial[elec])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


def plot_mean_data(trials, mean_target_data, mean_distract_data, mean_idle_data):
    """
    plot means for every stimulus for every electrode
    """
    elec_len = 8
    for i in range(elec_len):
        plt.figure()
        plt.suptitle('electorde #' + str(i+1))
        plt.plot(range(len(mean_target_data[0][i])), mean_target_data[0][i])
        plt.plot(range(len(mean_idle_data[0][i])), mean_idle_data[0][i])
        plt.plot(range(len(mean_distract_data[0][i])), mean_distract_data[0][i])
        plt.legend(['Target', 'Idle', 'Distract'], loc='lower right')
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()

def plot_down_sample(trials, down_sample_target, down_sample_idle):
    """
    plot means for every stimulus for every electrode
    """
    elec_len = 8
    for i in range(elec_len):
        plt.figure()
        plt.suptitle('electorde #' + str(i+1))
        # fig.set_size_inches(12, 8)
        plt.plot(range(len(down_sample_target[0][i])), down_sample_target[0][i])
        plt.plot(range(len(down_sample_idle[0][i])), down_sample_idle[0][i])
        plt.legend(['Target', 'Idle'], loc='lower right')
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()
