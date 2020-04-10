import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
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
    - transmissions = csr sparse matrix of i, j, lambda_ij
    - infection_probas[i]  = 1 - prod_{j: state==I} [1 - lambda_ij]
    We use prod_j  [1 - lambda_ij] = exp(sum_j log(1 - lambda_ij))
    """
    infected = (states == 1)
    infected_transmissions = transmissions.multiply(infected)
    infection_probas = 1 - np.exp(
        infected_transmissions.multiply(-1).log1p().sum(axis=1)
    )
    infection_probas = np.array(infection_probas).squeeze()
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
        - transmissions[t] = csr sparse matrix of i, j, lambda_ij(t)
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

    def sample_transmissions(self):
        raise NotImplementedError

    def generate_transmissions(self, T):
        self.transmissions = [self.sample_transmissions() for t in range(T)]

    def run(self, T, print_every=10):
        print("Generating transmissions")
        self.generate_transmissions(T)
        print("Running simulation")
        self.time_evolution(
            self.recover_probas, self.transmissions, print_every=print_every
        )

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
            np.fill_diagonal(proba_contact, 0.) # no contact with oneself
        self.proba_contact = proba_contact
        # expected number of contacts
        self.n_contacts = proba_contact.sum()/N
        # constant recovery proba
        self.recover_probas = mu*np.ones(N)
        super().__init__(initial_states, x_pos, y_pos)

    def sample_contacts(self):
        "contacts[i,j] = symmetric matrix, flag if i and j in contact"
        # sample only in lower triangular
        A = np.random.rand(self.N, self.N) < self.proba_contact
        L = np.tril(A, -1)
        # symmetrize
        contacts = np.maximum(L, L.T)
        return contacts

    def sample_transmissions(self):
        "transmissions = csr sparse matrix of i, j, lambda_ij"
        contacts = self.sample_contacts()
        i, j = np.where(contacts)
        # constant rate = lamb
        rates = self.lamb * np.ones(len(i))
        transmissions = coo_matrix(
            (rates, (i, j)), shape=(self.N, self.N)
        ).tocsr()
        # sanity check
        assert contacts.sum() == transmissions.nnz
        assert np.all(i != j)
        return transmissions


class NetworkModel(EpidemicModel):
    """
    Model:
    - graph = networkx undirected graph
    - mu = constant recovery proba
    - lamd = constant transmission rate (if in contact)
    - proba_contact = float, constant proba for all edges and time steps.
    At time step t, the edge ij is activated as a contact with proba_contact.
    So the contacts network is at each time a subgraph of the original graph.
    proba_contact = 1 corresponds to the fixed contacts network case.
    - initial_states = random patient zero by default
    - layout = spring layout by default

    You can also provide the initial_states and layout.
    """
    def __init__(self, graph, mu, lamb, proba_contact,
    initial_states = None, layout = None):
        self.graph = graph
        self.n_edges = graph.number_of_edges()
        self.mu = mu
        self.lamb = lamb
        self.proba_contact = proba_contact
        N = graph.number_of_nodes()
        # initial states : patient zero infected
        if initial_states is None:
            patient_zero = np.random.randint(N)
            initial_states = np.zeros(N)
            initial_states[patient_zero] = 1
        # positions
        if layout is None:
            print("Computing spring layout")
            layout = nx.spring_layout(graph)
        x_pos = np.array([layout[i][0] for i in graph.nodes])
        y_pos = np.array([layout[i][1] for i in graph.nodes])
        # expected number of contacts
        self.n_contacts = 2*self.n_edges*proba_contact/N
        # constant recovery proba
        self.recover_probas = mu*np.ones(N)
        super().__init__(initial_states, x_pos, y_pos)

    def sample_contacts(self):
        "contacts = list of i and j in contact"
        # each edge is selected with proba_contact
        selected = np.random.rand(self.n_edges) <= self.proba_contact
        contacts = [
            (i, j) for idx, (i, j) in enumerate(self.graph.edges)
            if selected[idx]
        ]
        # symmetrize
        contacts += [(j, i) for (i, j) in contacts]
        return contacts

    def sample_transmissions(self):
        "transmissions = csr sparse matrix of i, j, lambda_ij"
        contacts = self.sample_contacts()
        i = [i_ for (i_, j_) in contacts]
        j = [j_ for (i_, j_) in contacts]
        # constant rate = lamb
        rates = self.lamb * np.ones(len(i))
        transmissions = coo_matrix(
            (rates, (i, j)), shape=(self.N, self.N)
        ).tocsr()
        # sanity check
        assert len(contacts) == transmissions.nnz
        assert np.all(i != j)
        return transmissions
