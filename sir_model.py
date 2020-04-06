import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import logging
logger = logging.getLogger(__name__)

STATES = ["S", "I", "R"]

def get_infection_probas(states, transmissions):
    """
    - states[i] = state of i
    - transmissions = array/list of i, j, lambda_ij
    - infection_probas[i]  = 1 - prod_{j: state==I} [1 - lambda_ij]
    """
    infected = (states == 1)
    N = len(states)
    infection_probas = np.zeros(N)
    for i in range(N):
        rates = np.array([
            rate for i0, j, rate in transmissions
            if i0 == i and infected[j]
        ])
        infection_probas[i] = 1 - np.prod(1 - rates)
    return infection_probas


def propagate(current_states, infection_probas, recover_probas):
    """
    - current_states[i] = state of i
    - infection_probas[i]  = proba that i get infected (if susceptible)
    - recover_probas[i] = proba that i recovers (if infected)
    """
    next_states = np.zeros_like(current_states)
    for i, state in enumerate(current_states):
        if (state == 0):
            infected = np.random.rand() < infection_probas[i]
            next_states[i] = 1 if infected else 0
        elif (state == 1):
            recovered = np.random.rand() < recover_probas[i]
            next_states[i] = 2 if recovered else 1
        else:
            next_states[i] = 2
    return next_states


class EpidemicModel():
    def __init__(self, initial_states, x_pos, y_pos):
        assert len(x_pos) == len(y_pos) == len(initial_states)
        self.N = len(initial_states)
        self.initial_states = initial_states
        self.x_pos = x_pos
        self.y_pos = y_pos

    def time_evolution(self, recover_probas, transmissions, print_every=10):
        """Run the simulation where
        - recover_probas[i] = mu_i time-independent
        - transmissions[t] = list of t, i, j, lambda_ij(t)
        - states[t, i] = state of i at time t
        """
        # initialize states
        T = len(transmissions)
        states = np.empty((T + 1, self.N))
        states[:] = np.nan
        states[0] = self.initial_states
        # iterate over time steps
        for t in range(T):
            if (t % print_every == 0):
                print(f"t = {t} / {T}")
            infection_probas = get_infection_probas(states[t], transmissions[t])
            states[t+1] = propagate(states[t], infection_probas, recover_probas)
        self.states = states

    def plot(self, t):
        fig, ax = plt.subplots(1, 1, figsize = (5,5))
        for idx, state in enumerate(STATES):
            ind = np.where(self.states[t] == idx)
            ax.scatter(self.x_pos[ind], self.y_pos[ind], label=state)
        ax.set(title="t = %d" % t)
        ax.legend()

    def get_counts(self):
        counts = {
            state: (self.states == idx).sum(axis=1)
            for idx, state in enumerate(STATES)
        }
        return pd.DataFrame(counts)

class ProximityModel(EpidemicModel):
    """
    Model:
    - N = population
    - mu = constant recovery proba
    - lamd = constant transmission rate (if in contact)
    - proba of contact = np.exp(-distance / scale)
    """
    def __init__(self, N, scale, mu, lamb):
        self.scale = scale
        self.mu = mu
        self.lamb = lamb
        # initial states : patient zero infected
        patient_zero = np.random.randint(N)
        initial_states = np.zeros(N)
        initial_states[patient_zero] = 1
        # positions
        pos = np.sqrt(N)*np.random.rand(N, 2)
        x_pos, y_pos = pos.T
        # proba of contact = np.exp(-distance / scale)
        distance = squareform(pdist(pos))
        proba_contact = np.exp(-distance / scale)
        np.fill_diagonal(proba_contact, False) # no contact with oneself
        self.proba_contact = proba_contact
        # expected number of contacts
        self.n_contacts = proba_contact.sum()/N
        # constant recovery proba
        self.recover_probas = mu*np.ones(N)
        super().__init__(initial_states, x_pos, y_pos)

    def sample_contacts(self):
        "contacts[i,j] = if i and j in contact"
        contacts = np.random.rand(self.N, self.N) < self.proba_contact
        return contacts

    def sample_transmissions(self):
        "transmissions = list of t, i, j, lambda_ij"
        contacts = self.sample_contacts()
        i, j = np.where(contacts)
        # constant rate = lamb
        rates = self.lamb * np.ones(len(i))
        transmissions = list(zip(i, j, rates))
        # sanity check
        assert contacts.sum() == len(transmissions)
        assert np.all(i != j)
        return transmissions

    def run(self, T):
        print("Generating transmissions")
        self.transmissions = [self.sample_transmissions() for t in range(T)]
        print("Running simulation")
        self.time_evolution(self.recover_probas, self.transmissions)
