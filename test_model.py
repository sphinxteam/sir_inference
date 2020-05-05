import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import pickle
from sir_model import indicator, STATES


def compute_averages(model, n_run, times, fname=None, print_every=100):
    """
    Computes the Monte Carlo frequencies and average estimated probabilites.
    - model : EpidemicModel instance to generate the SIR simulation
    - infer : BaseInference instance to estimate the probabilites
    - n_run : number of Monte Carlo runs
    - times : times at which to look at
    - fname : if given saves the average states in a .p file with pickle
    """
    n_times = len(times)
    t_runs = np.zeros(n_run)
    t_max = max(times)+1
    states = np.zeros((n_run, n_times, model.N, 3))
    transmissions = model.transmissions[:t_max]
    recover_probas = model.recover_probas
    # monte carlo runs
    for n in range(n_run):
        if (n % print_every == 0):
            print(f"n = {n} / {n_run}")
        model.time_evolution(recover_probas, transmissions, print_every=0)
        states[n] = indicator(model.states[times])
    # <.> over monte carlo runs
    avg_states = states.mean(axis=0)  # avg_states[t,i,s] = < q_i(t) == s >
    if fname is not None:
        pickle.dump(avg_states, open(fname+'_avg_states.p', "wb"))
    return avg_states


def generate_scatterplot(avg_states, probas, times):
    """
    Scatterplot of the Monte Carlo frequencies vs the estimated probabilites.
    - model : EpidemicModel instance to generate the SIR simulation
    - infer : BaseInference instance to estimate the probabilites
    - n_run : number of Monte Carlo runs
    - times : times at which to look at
    """
    n_times = len(times)
    fig, axs = plt.subplots(
        n_times, 3, figsize=(4*3, 4*n_times),
        sharex=True, sharey=True, squeeze=False
    )
    for t, row in enumerate(axs):
        for s, ax in enumerate(row):
            ax.plot([0, 1], [0, 1])
            ax.scatter(probas[t, :, s], avg_states[t, :, s], color='red')
            ax.set(
                xlabel="average $P_s^i(t)$",
                ylabel="frequency of $q_i(t) = s$",
                title=f"{STATES[s]}    t={times[t]}", xlim=(0, 1), ylim=(0, 1)
            )
    fig.tight_layout()
