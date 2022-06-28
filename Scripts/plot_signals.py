import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from plot_title import make_plot_title

config = OmegaConf.load("config.yaml")
samp_freq = config.sample_frequency
color_codes = [c[1] for c in config.colors.items()]


def plot_record(record, filename):
    # Set up x-axis in time domain
    channels = record.shape[1]
    datapoints = record.shape[0]
    x = np.linspace(0, datapoints / samp_freq, datapoints)

    # Compose plot
    fig = plt.figure(figsize=(12, 8))
    color = 0

    for i in range(0, channels):
        if "ICA" in filename:
            y = i / 10
            plt.text(-1, -y - 0.02, record.columns[i], fontsize="small", ha="right")
        else:
            y = i * 20
            plt.text(-1, -y - 3, record.columns[i], fontsize="small", ha="right")
        # if color == len(color_codes):
        color = 0
        plt.plot(x, record.iloc[:, i] - y, color=color_codes[color], linewidth=0.75)
        color += 1

    plt.title(make_plot_title(filename), fontsize="x-large")
    plt.xlabel("Time [s]")
    plt.yticks([])
    plt.box(False)
    plt.grid(color="#D6D6D6")
    fig.tight_layout()

    return fig
