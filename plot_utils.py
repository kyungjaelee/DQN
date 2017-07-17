import numpy as np
import matplotlib.pyplot as plt
import math
from cycler import cycler

rcParams = plt.matplotlib.rcParams


def get_prop_cycle():
    prop_cycler = rcParams['axes.prop_cycle']
    if prop_cycler is None and 'axes.color_cycle' in rcParams:
        clist = rcParams['axes.color_cycle']
        prop_cycler = cycler('color', clist)
    return prop_cycler


def shadow_plot(x, y, **kwargs):
    """
    By default, the plotted shadow subtends between the 5% and 95% of the data.
    :param x: 
    :param y: 
    :param kwargs: 
    :return: 
    """
    n_samples = y.size      # Number of samples
    smooth_factor = 0.5     # Default smooth factor
    label = None
    semilogy = False
    clr = None
    shadow_std = False      # Whether to subtend the shadow between -std and +std or not.

    ax = plt.gca()
    if get_prop_cycle() is None:
        cmap = plt.cm.get_cmap(name="Vega20", lut=64)
        ax.set_prop_cycle(cycler('color', cmap(range(64))))

    for key, value in kwargs.items():
        if key == "label":
            label = value
        elif key == "smooth":  # Smooth factor
            smooth_factor = value
            if smooth_factor < 0 or smooth_factor > 1:
                raise ValueError("The smooth factor must lie between 0 and 1.")
        elif key == "semilogy":
            semilogy = value
            if not isinstance(semilogy, bool):
                raise ValueError("semilogy can only be True or False.")
        elif key == "color":
            clr = value
        elif key == "shadowstd":
            if not isinstance(semilogy, bool):
                raise ValueError("shadowstd can only be True or False.")
            shadow_std = value

    n = int(math.ceil(smooth_factor * 0.25 * n_samples))  # Window size for averaging (at most, 25% of all samples)
    mu = np.full(n_samples, np.nan)
    sigma = np.full(n_samples, np.nan)
    perc5 = np.full(n_samples, np.nan)
    perc95 = np.full(n_samples, np.nan)
    for s in range(n_samples):
        s0 = max(s - n + 1, 0)
        mu[s] = np.mean(y[s0:s + 1])
        sigma[s] = np.std(y[s0:s + 1])
        perc5[s] = np.percentile(y[s0:s + 1], 5)
        perc95[s] = np.percentile(y[s0:s + 1], 95)
    assert np.all(sigma >= 0), "Negative standard deviation"

    if semilogy:
        base_line, = ax.semilogy(x, mu, lw=2, label=label, color=clr)
    else:
        base_line, = ax.plot(x, mu, lw=2, label=label, color=clr)

    if shadow_std:
        higher_border = mu + sigma
        lower_border = mu - sigma
    else:
        higher_border = perc95
        lower_border = perc5

    if semilogy:
        # If the lower border of the shadowed region is 0 or negative, clip its minimum value. The minimum value shall
        # be equidistant from the average in log scale.
        factor = higher_border / mu
        lower_border = np.clip(lower_border, mu / factor, mu)

    ax.fill_between(x, higher_border, lower_border, facecolor=base_line.get_color(), alpha=0.2)
