import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STATES = ["S", "I", "R"]


def get_infection_probas(probas, transmissions):
    """
    - probas[i,s] = P_s^i(t)
    - transmissions = sparse matrix of i, j, lambda_ij
    - infection_probas[i]  = sum_j lambda_ij P_I^j(t)
    """
    N = probas.shape[0]
    infection_probas = np.zeros(N)
    for i in range(N):
        contact_with_i = transmissions.toarray()[i]
        rates = np.array([
            probas[i, 1]*rate for rate in contact_with_i[contact_with_i > 0]  # use lambda_ij > 0 to identify people in contact
        ])
        infection_probas[i] = rates.sum()
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


class InferenceModel():
    def __init__(self, initial_probas, x_pos, y_pos):
        assert initial_probas.shape[1] == 3
        assert len(x_pos) == len(y_pos) == initial_probas.shape[0]
        self.N = len(initial_probas)
        self.initial_probas = initial_probas
        self.x_pos = x_pos
        self.y_pos = y_pos

    def time_evolution(self, recover_probas, transmissions, print_every=10):
        """Run the simulation where
        - recover_probas[i] = mu_i time-independent
        - transmissions[t] = list of t, (i, j, lambda_ij) where (i, j, lambda_ij) is in a sparse matrix
        - probas[t, i, s] = state of i at time t
        """
        # initialize states
        T = len(transmissions)
        probas = np.zeros((T + 1, self.N, 3))
        probas[0] = self.initial_probas
        # iterate over time steps
        for t in range(T):
            if (t % print_every == 0):
                print(f"t = {t} / {T}")
            infection_probas = get_infection_probas(probas[t], transmissions[t])
            probas[t+1] = propagate(probas[t], infection_probas, recover_probas)
        self.probas = probas
        self.states = probas.argmax(axis=2)

    def plot_states(self, t):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for idx, state in enumerate(STATES):
            ind = np.where(self.states[t] == idx)
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

    def get_counts(self):
        counts = self.probas.sum(axis=1)
        return pd.DataFrame(counts, columns=STATES)
