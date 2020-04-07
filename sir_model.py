import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import logging
logger = logging.getLogger(__name__)

STATES = ["S", "I", "R"]

def get_dummies(states):
    probas = np.zeros(states.shape + (3,))
    for s in [0,1,2]:
        probas[:,:,s] = (states==s)*1
    assert np.all(probas.argmax(axis=2) == states)
    return probas


def get_infection_probas(states, transmissions):
    """
    - states[i] = state of i
    - transmissions = sparse matrix of i, j, lambda_ij
    - infection_probas[i]  = 1 - prod_{j: state==I} [1 - lambda_ij]

    trick prod_j (1 - A_ij) = exp(sum_j ln(1-A_ij))
    """
    infected = (states == 1)
    N = len(states)
    infected_transmissions = transmissions.multiply(infected)  # element-wise multiplication for masking
    infection_probas = 1 - np.exp(np.sum(np.log1p(infected_transmissions.multiply(-1)), axis=1))
    infection_probas = np.squeeze(np.asarray(infection_probas))
    assert len(infection_probas) == N  # sanity check
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
        - transmissions[t] = list of t, (i, j, lambda_ij) where (i, j, lambda_ij) is in a sparse matrice
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
        self.probas = get_dummies(states)

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
    - proba_contact = np.exp(-distance / scale)
    - initial_states = random patient zero
    - x_pos, y_pos = random uniform

    You can also provide the initial_states, x_pos, y_pos or proba_contact.
    """
    def __init__(self, N, scale, mu, lamb,
    initial_states = None, x_pos = None, y_pos = None, proba_contact = None):
        self.scale = scale
        self.mu = mu
        self.lamb = lamb
        # initial states : patient zero infected
        if initial_states is None:
            patient_zero = np.random.randint(N)
            initial_states = np.zeros(N)
            initial_states[patient_zero] = 1
        # positions
        x_pos = np.sqrt(N)*np.random.rand(N) if x_pos is None else x_pos
        y_pos = np.sqrt(N)*np.random.rand(N) if y_pos is None else y_pos
        if proba_contact is None:
            # proba of contact = np.exp(-distance / scale)
            pos = np.array([x_pos, y_pos]).T
            assert pos.shape == (N, 2)
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
        "transmissions = list of t, (i, j, lambda_ij) where (i, j, lambda_ij) is in a sparse matrix"
        contacts = self.sample_contacts()
        i, j = np.where(contacts)
        # constant rate = lamb
        rates = self.lamb * np.ones(len(i))
        transmissions = coo_matrix((rates, (i, j)), shape=(self.N, self.N))
        # sanity check
        assert contacts.sum() == transmissions.nnz  # number of stored values
        assert np.all(i != j)
        return transmissions

    def generate_transmissions(self, T):
        self.transmissions = [self.sample_transmissions() for t in range(T)]

    def run(self, T, print_every=10):
        print("Generating transmissions")
        self.generate_transmissions(T)
        print("Running simulation")
        self.time_evolution(
            self.recover_probas, self.transmissions, print_every=print_every
        )
