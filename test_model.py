import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


def indicator(states):
    probas = np.zeros(states.shape + (3,))
    for s in [0,1,2]:
        probas[..., s] = (states==s)*1
    assert np.all(probas.argmax(axis = -1) == states)
    return probas


def compute_averages(model, infer, n_run, times):
    """
    Computes the Monte Carlo frequencies and average estimated probabilites.
    - model : EpidemicModel instance to generate the SIR simulation
    - infer : BaseInference instance to estimate the probabilites
    - n_run : number of Monte Carlo runs
    - times : times at which to look at
    """
    # complete observation of the initial states
    infer.initial_probas = indicator(model.initial_states)
    # storage, states (as indicator) and probas
    n_times = len(times)
    t_runs = np.zeros(n_run)
    states = np.zeros((n_run, n_times, model.N, 3))
    probas = np.zeros((n_run, n_times, model.N, 3))
    # monte carlo runs
    for n in range(n_run):
        if (n % 100 == 0): print(f"n = {n} / {n_run}")
        t_max = max(times) + 1
        model.run(t_max, print_every=0)
        states[n] = indicator(model.states[times])
        tic = time()
        infer.time_evolution(model.recover_probas, model.transmissions, print_every=0)
        t_runs[n] = time() - tic
        probas[n] = infer.probas[times]
    print(
        f"run times from {t_runs.min():.1e}s to {t_runs.max():.1e}s "
        f"median {np.median(t_runs):.1e}s"
    )
    # <.> over monte carlo runs
    avg_states = states.mean(axis=0) # avg_states[t,i,s] = < q_i(t) == s >
    avg_probas = probas.mean(axis=0) # avg_probas[t,i,s] = < p_i^s[t] >
    return avg_states, avg_probas


def generate_scatterplot(model, infer, n_run, times):
    """
    Scatterplot of the Monte Carlo frequencies vs the estimated probabilites.
    - model : EpidemicModel instance to generate the SIR simulation
    - infer : BaseInference instance to estimate the probabilites
    - n_run : number of Monte Carlo runs
    - times : times at which to look at
    """
    avg_states, avg_probas = compute_averages(model, infer, n_run, times)
    n_times = len(times)
    # scatterplot
    STATES = "SIR"
    fig, axs = plt.subplots(
        n_times, 3, figsize=(4*3, 4*n_times),
        sharex=True, sharey=True, squeeze=False
    )
    for t, row in enumerate(axs):
        for s, ax in enumerate(row):
            ax.plot([0, 1], [0, 1])
            ax.scatter(avg_probas[t, :, s], avg_states[t, :, s])
            ax.set(
                xlabel="average $P_s^i(t)$",
                ylabel="frequency of $q_i(t) = s$",
                title=f"{STATES[s]}    t={times[t]}", xlim=(0, 1), ylim=(0, 1)
            )
    fig.tight_layout()
