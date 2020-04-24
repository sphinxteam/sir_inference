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


def compute_averages(model, n_run, times,fname=None):
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
    states = np.zeros((n_run, n_times, model.N, 3))
    probas = np.zeros((n_run, n_times, model.N, 3))
    # monte carlo runs
    for n in range(n_run):
        if (n % 100 == 0): print(f"n = {n} / {n_run}")
        t_max = max(times) + 1
        model.run(t_max, print_every=0)
        states[n] = indicator(model.states[times])
        tic = time()
        t_runs[n] = time() - tic
        
    print(
        f"run times from {t_runs.min():.1e}s to {t_runs.max():.1e}s "
        f"median {np.median(t_runs):.1e}s"
    )
    # <.> over monte carlo runs
    avg_states = states.mean(axis=0) # avg_states[t,i,s] = < q_i(t) == s >
    if fname is not None:
        pickle.dump( avg_states, open( fname+'_avg_states.p', "wb" ) )
    return avg_states


def generate_scatterplot(model, infer, n_run, times,axs=None,fname=None):
    """
    Scatterplot of the Monte Carlo frequencies vs the estimated probabilites.
    - model : EpidemicModel instance to generate the SIR simulation
    - infer : BaseInference instance to estimate the probabilites
    - n_run : number of Monte Carlo runs
    - times : times at which to look at
    - fname : if given saves the average states in a .p file with pickle
    """
    avg_states = compute_averages(model, n_run, times,fname=fname)
    n_times = len(times)
    
    infer.initial_probas = indicator(model.initial_states)
    infer.time_evolution(model.recover_probas, model.transmissions, print_every=0)
    probas = infer.probas[times]
    
    # scatterplot
    STATES = "SIR"
    if axs is None:
        fig, axs = plt.subplots(
            n_times, 3, figsize=(4*3, 4*n_times),
            sharex=True, sharey=True, squeeze=False
        )
        
    if n_times>1:
        for t, row in enumerate(axs):
            for s, ax in enumerate(row):
                p=ax.plot([0, 1], [0, 1])
                p=ax.scatter(probas[t, :, s], avg_states[t, :, s])
                ax.set(
                    xlabel="average $P_s^i(t)$",
                    ylabel="frequency of $q_i(t) = s$",
                    title=f"{STATES[s]}    t={times[t]}", xlim=(0, 1), ylim=(0, 1)
                )
    else:
        if n_times==1:
            t=0
            for s, ax in enumerate(axs):
                p=ax.plot([0, 1], [0, 1])
                p=ax.scatter(probas[t, :, s], avg_states[t, :, s])
                ax.set(
                    xlabel="average $P_s^i(t)$",
                    ylabel="frequency of $q_i(t) = s$",
                    title=f"{STATES[s]}    t={times[t]}", xlim=(0, 1), ylim=(0, 1)
                )
    if axs is None:
        fig.tight_layout()
    return p
