import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist, squareform
import logging
logger = logging.getLogger(__name__)

STATES = ["S", "I", "R"]


def indicator(states):
    probas = np.zeros(states.shape + (3,))
    for s in [0,1,2]:
        probas[..., s] = (states==s)*1
    assert np.all(probas.argmax(axis = -1) == states)
    return probas


def frequency(states, verbose=True):
    "Generate initial proba according to the frequencies of states"
    freqs = [np.mean(states==s) for s in [0,1,2]]
    if verbose:
        print("freqs = ", freqs)
    N = len(states)
    initial_probas = np.broadcast_to(freqs, (N, 3)).copy()
    return initial_probas


def patient_zeros_states(N, N_patient_zero):
    states = np.zeros(N)
    patient_zero = np.random.choice(N, N_patient_zero, replace=False)
    states[patient_zero] = 1
    return states


def random_individuals(N, n_obs):
    return np.random.choice(N, n_obs, replace=False)


def infected_individuals(states, n_obs):
    infected, = np.where(states == 1)
    if len(infected) < n_obs:
        print(
            f"WARNING only {len(infected)} infected "
            f"cannot return n_obs={n_obs} observations"
        )
        return infected
    return np.random.choice(infected, n_obs, replace=False)


def random_observations(model, tests):
    """
    Observations given by random sampling of the population.

    Parameters
    ----------
    - model : EpidemicModel instance to gives the states
    - tests : dict
        n_test = tests[t_test] number of random tests done at t=t_test

    Returns
    -------
    - observations : list of dict(i=i, s=s, t_test=t_test) observations
    """
    observations = []
    for t_test, n_test in tests.items():
        tested = random_individuals(model.N, n_test)
        for i in tested:
            obs = dict(i=i, t_test=t_test, s=model.states[t_test, i])
            observations.append(obs)
    return observations


def infected_observations(model, t_test, n_test):
    """
    Observations corresponding to n_test infected individuals at t=t_test.

    Parameters
    ----------
    - model : EpidemicModel instance to gives the states
    - t_test : int
    - n_test : int

    Returns
    -------
    - observations : list of dict(i=i, s=s, t_test=t_test) observations
    """
    infected = infected_individuals(model.states[t_test], n_test)
    observations = [dict(i=i, t_test=t_test, s=1) for i in infected]
    return observations


def get_infection_probas(states, transmissions):
    """
    - states[i] = state of i
    - transmissions = csr sparse matrix of i, j, lambda_ij
    - infection_probas[i]  = 1 - prod_{j: s[j]==I} [1 - lambda_ij]
    We use prod_{j:I}  [1 - lambda_ij] = exp(
        sum_j log(1 - lambda_ij) (s[j]==I)
    )
    """
    infected = (states == 1)
    infection_probas = 1 - np.exp(
        transmissions.multiply(-1).log1p().dot(infected)
    )
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
        states = np.zeros((T + 1, self.N), dtype=int)
        states[0] = self.initial_states
        # iterate over time steps
        for t in range(T):
            if print_every and (t % print_every == 0):
                print(f"t = {t} / {T}")
            infection_probas = get_infection_probas(states[t], transmissions[t])
            states[t+1] = propagate(states[t], infection_probas, recover_probas)
        self.states = states
        self.probas = indicator(states)
        self.recover_probas = recover_probas
        self.transmissions = transmissions

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
        if print_every:
            print("Generating transmissions")
        self.generate_transmissions(T)
        if print_every:
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


def read_ferretti_data(csv_file, lamb):
    N = 10000
    df = pd.read_csv(csv_file)
    assert N-1 == df["ID"].max() == df["ID_2"].max()
    tmax = df["time"].max()
    transmissions = []
    for t in range(tmax):
        sub_data = df.query(f"time=={t}")
        i, j = sub_data["ID"], sub_data["ID_2"]
        rates = lamb*np.ones_like(i)
        transmissions.append(
            csr_matrix((rates, (i, j)), shape=(N, N))
        )
    return transmissions


def proximity_model(N, N_patient_zero, scale, mu, lamb, t_max, seed):
    print("Using ProximityModel")
    np.random.seed(seed)
    initial_states = patient_zeros_states(N, N_patient_zero)
    model = ProximityModel(N, scale, mu, lamb, initial_states)
    print("expected number of contacts %.1f" % model.n_contacts)
    model.run(t_max, print_every=100)
    return model


def ferretti_model(N_patient_zero=10, mu=1/15, lamb=0.02, seed=123):
    print("Using Ferretti transmissions")
    N = 10000
    transmissions = read_ferretti_data("all_interaction_10000.csv", lamb=lamb)
    initial_states = patient_zeros_states(N, N_patient_zero)
    # random x_pos, y_pos
    x_pos = np.random.rand(N)
    y_pos = np.random.rand(N)
    model = EpidemicModel(initial_states=initial_states, x_pos=x_pos, y_pos=y_pos)
    recover_probas = mu*np.ones(N)
    model.time_evolution(recover_probas, transmissions, print_every=100)
    return model
