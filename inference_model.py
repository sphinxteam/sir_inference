import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

STATES = ["S", "I", "R"]


def get_infection_probas(probas, transmissions):
    """
    - probas[i,s] = P_s^i(t)
    - transmissions = csr sparse matrix of i, j, lambda_ij
    - infection_probas[i]  = sum_j lambda_ij P_I^j(t)
    """
    infection_probas = transmissions.dot(probas[:, 1])
    return infection_probas


def propagate(probas, infection_probas, recover_probas):
    """
    - probas[i,s] = P_s^i(t)
    - infection_probas[i]  = proba that i get infected (if susceptible)
    - recover_probas[i] = proba that i recovers (if infected)
    """
    next_probas = np.zeros_like(probas)
    next_probas[:, 0] = probas[:, 0]*(1 - infection_probas)
    next_probas[:, 1] = probas[:, 1]*(1 - recover_probas) + probas[:, 0]*infection_probas
    next_probas[:, 2] = probas[:, 2] + probas[:, 1]*recover_probas
    return next_probas


def reset_probas(t, probas, observations):
    """
    Reset probas[t] according to observations
    - observations = list of dict(i=i, s=s, t=t) observations at t_obs=t
    If s=I, the observation must also give t_I the infection time
    - probas[t, i, s] = P_s^i(t)
    """
    for obs in observations:
        if (obs["s"] == 0) and (t <= obs["t"]):
            probas[t, obs["i"], :] = [1, 0, 0]  # p_i^S = 1
        if (obs["s"] == 1) and (obs["t_I"] <= t) and (t <= obs["t"]):
            probas[t, obs["i"], :] = [0, 1, 0]  # p_i^I = 1
        if (obs["s"] == 2) and (t >= obs["t"]):
            probas[t, obs["i"], :] = [0, 0, 1]  # p_i^R = 1


class InferenceModel():
    def __init__(self, initial_probas, x_pos, y_pos):
        assert initial_probas.shape[1] == 3
        assert len(x_pos) == len(y_pos) == initial_probas.shape[0]
        self.N = len(initial_probas)
        self.initial_probas = initial_probas
        self.x_pos = x_pos
        self.y_pos = y_pos

    def time_evolution(self, recover_probas, transmissions, observations=[], print_every=10):
        """
        Run the probability evolution where
        - recover_probas[i] = mu_i time-independent
        - transmissions[t] = csr sparse matrix of i, j, lambda_ij(t)
        - observations = list of dict(i=i, s=s, t=t) observations at t_obs=t
        If s=I, the observation must also give t_I the infection time
        - probas[t, i, s] = P_s^i(t)
        """
        # initialize states
        T = len(transmissions)
        probas = np.zeros((T + 1, self.N, 3))
        probas[0] = self.initial_probas
        # iterate over time steps
        for t in range(T):
            if (t % print_every == 0):
                print(f"t = {t} / {T}")
            reset_probas(t, probas, observations)
            infection_probas = get_infection_probas(probas[t], transmissions[t])
            probas[t+1] = propagate(probas[t], infection_probas, recover_probas)
        self.probas = probas
        self.states = probas.argmax(axis=2)

    def plot_states(self, t):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for s, state in enumerate(STATES):
            ind = np.where(self.states[t] == s)
            ax.scatter(self.x_pos[ind], self.y_pos[ind], label=state)
        ax.set(title="t = %d" % t)
        ax.legend()

    def plot_probas(self, t):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for s, (ax, state) in enumerate(zip(axs, STATES)):
            ax.scatter(self.x_pos, self.y_pos, c=self.probas[t, :, s],
                       cmap="Blues", vmin=0, vmax=1)
            ax.set(title=state)
        fig.tight_layout()

    def plot_probas_obs(self, t, model, observations, t_start):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for s, (ax, state) in enumerate(zip(axs, STATES)):
            ax.scatter(self.x_pos, self.y_pos, c=self.probas[t, :, s],
                       cmap="Blues", vmin=0, vmax=1)
            ind, = np.where(model.states[t_start + t] == s)
            observed = [
                obs["i"] for obs in observations
                if (obs["t_test"] == t_start + t) and (obs["s"] == s)
            ]
            unobserved = [i for i in ind if i not in observed]
            ax.scatter(model.x_pos[observed], model.y_pos[observed],
                facecolors='none', edgecolors='r')
            ax.scatter(model.x_pos[unobserved], model.y_pos[unobserved],
                facecolors='none', edgecolors='g')
            ax.set(title=state)
        fig.tight_layout()


    def get_counts(self):
        counts = self.probas.sum(axis=1)
        return pd.DataFrame(counts, columns=STATES)
