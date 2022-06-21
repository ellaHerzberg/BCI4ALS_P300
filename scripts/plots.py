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


def plot_down_sample(trials, down_sample_target, down_sample_idle):
    """
    plot means for every stimulus for every electrode
    """
    elec_len = len(trials[0][0])
    for i in range(elec_len):
        plt.figure()
        fig, axs = plt.subplots(3)
        fig.suptitle('electorde #' + str(i))
        # fig.set_size_inches(12, 8)
        axs[0].plot(range(len(down_sample_target[0][i])), down_sample_target[0][i])
        axs[0].set_title('Target')
        axs[1].plot(range(len(down_sample_idle[0][i])), down_sample_idle[0][i])
        axs[1].set_title('Idle')
        plt.xlabel("Time")
        for ax in axs.flat:
            ax.set(ylabel='Frequency')
        plt.show()
