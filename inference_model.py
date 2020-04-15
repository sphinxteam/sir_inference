import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt

STATES = ["S", "I", "R"]


def infos_csr(t, name, array):
    return dict(
        t=t, name=name, shape=array.shape, nnz=array.nnz,
        nan=(array==np.nan).sum(), min=array.min(), max=array.max()
    )


def infos_array(t, name, array):
    return dict(
        t=t, name=name, shape=array.shape, nnz=(array!=0).sum(),
        nan=(array==np.nan).sum(), min=array.min(), max=array.max()
    )


def zero_csr(N):
    i = j = data = []
    return csr_matrix((data, (i, j)), shape=(N, N))


def vector_csr(v):
    """
    - A = csr matrix A_ij = v_j
    """
    A = np.tile(v, (len(v), 1))
    return csr_matrix(A)


def initial_messages(initial_probas):
    """
    - initial_probas[j, s] = P_s^j(t=0)
    - kappa = csr sparse matrix of i, j, kappa_ij(0) = 0
    - P_bar = csr sparse matrix of i, j, P_bar_ij(0) = 1 - P_j^S(0)
    - phi = csr sparse matrix of i, j, phi_ij(0) = P_j^I(0)
    """
    N = initial_probas.shape[0]
    kappa = zero_csr(N)
    P_bar = vector_csr(1 - initial_probas[:,0])
    phi = vector_csr(initial_probas[:,1])
    return kappa, P_bar, phi


def sum_contacts(A, contacts):
    """
    - A = csr sparse matrix of i, k, A_ik
    - contacts = csr sparse matrix of i, k in contact
    - a_i = sum_{k in i} A_ik  = sum_k A_ik contacts_ik
    """
    ones = np.ones(A.shape[0])
    a = A.multiply(contacts).dot(ones)
    return a


def sum_messages(A, contacts):
    """
    - A = csr sparse matrix of j, k, A_jk
    - contacts = csr sparse matrix of j, k in contact
    - B_ij = sum_{k in j\i} A_jk  = sum_k A_jk contacts_jk (k != i)
    """
    unequal = 1 - np.identity(A.shape[0])
    B_transpose = A.multiply(contacts).dot(unequal)
    B = csr_matrix(B_transpose).transpose()
    return B


def get_infection_messages(kappa, contacts, kappa_next, full_contacts):
    """
    Parameters
    ----------
    - kappa = csr sparse matrix of i, j, kappa_ij(t)
    - contacts = csr sparse matrix of i, j in contact at t
    - kappa_next = csr sparse matrix of i, j, kappa_ij(t+1)
    - full_contacts = csr sparse matrix of i, j in contact at any time

    Returns
    -------
    - infection_messages = csr sparse matrix of i, j, rho_ij
    rho_ij = 1 - exp(
        sum_{k in j(t)\i} ln[1 - kappa_jk(t+1)] - ln[1 - kappa_jk(t)]
    )
    """
    A = sum_messages(kappa.multiply(-1).log1p(), contacts)
    A_next = sum_messages(kappa_next.multiply(-1).log1p(), contacts)
    dA = A_next - A
    infection_messages = dA.expm1().multiply(-1)
    infection_messages = full_contacts.multiply(infection_messages)
    return infection_messages


def update_messages(kappa, P_bar, phi, transmissions, recover_probas, full_contacts):
    """
    Parameters
    ----------
    - kappa = csr sparse matrix of i, j, kappa_ij(t)
    - P_bar = csr sparse matrix of i, j, P_bar_ij(t)
    - phi = csr sparse matrix of i, j, phi_ij(t)
    - transmissions = csr sparse matrix of i, j, kappa_ij(t)
    - recover_probas[i] = mu_i
    - full_contacts = csr sparse matrix of i, j in contact at any time

    Returns
    -------
    - kappa_next = csr sparse matrix of i, j, kappa_ij(t+1)
    - P_bar_next = csr sparse matrix of i, j, P_bar_ij(t+1)
    - phi_next = csr sparse matrix of i, j, phi_ij(t+1)
    """
    kappa_next = kappa + transmissions.multiply(phi)
    contacts = (transmissions != 0)
    rho = get_infection_messages(kappa, contacts, kappa_next, full_contacts)
    delta_P_bar = rho - rho.multiply(P_bar)
    P_bar_next = P_bar + delta_P_bar
    phi_next = (
        (phi - phi.multiply(transmissions)).multiply(1 - recover_probas)
        + delta_P_bar
    )
    return kappa_next, P_bar_next, phi_next


def get_infection_probas_dmp(kappa, contacts, kappa_next):
    """
    Parameters
    ----------
    - kappa = csr sparse matrix of i, j, kappa_ij(t)
    - contacts = csr sparse matrix of i, j in contact at t
    - kappa_next = csr sparse matrix of i, j, kappa_ij(t+1)

    Returns
    -------
    - infection_probas = array
    rho_i = 1 - exp(
        sum_{k in i(t)} ln[1 - kappa_ik(t+1)] - ln[1 - kappa_ik(t)]
    )
    """
    a = sum_contacts(kappa.multiply(-1).log1p(), contacts)
    a_next = sum_contacts(kappa_next.multiply(-1).log1p(), contacts)
    da = a_next - a
    infection_probas = -np.expm1(da)
    return infection_probas


############### Mean field ##############################
def get_infection_probas_mean_field(probas, transmissions):
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
    - probas_next[i, s] = P_s^i(t+1)
    """
    probas_next = np.zeros_like(probas)
    probas_next[:, 0] = probas[:, 0]*(1 - infection_probas)
    probas_next[:, 1] = probas[:, 1]*(1 - recover_probas) + probas[:, 0]*infection_probas
    probas_next[:, 2] = probas[:, 2] + probas[:, 1]*recover_probas
    return probas_next


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


class BaseInference():
    def __init__(self, initial_probas, x_pos, y_pos):
        assert initial_probas.shape[1] == 3
        assert len(x_pos) == len(y_pos) == initial_probas.shape[0]
        self.N = len(initial_probas)
        self.initial_probas = initial_probas
        self.x_pos = x_pos
        self.y_pos = y_pos

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


def get_full_contacts(transmissions):
    """
    - transmissions[t] = csr sparse matrix of i, j, lambda_ij
    - full_contacts = csr sparse matrix of i, j in contact at any time t
    """
    full_contacts = (transmissions[0] != 0)
    for tr in transmissions:
        contacts = (tr != 0)
        full_contacts = full_contacts.maximum(contacts)
    return full_contacts


class MeanField(BaseInference):

    def time_evolution(self, recover_probas, transmissions, observations=[], print_every=10):
        """
        Run the probability evolution from t = 0 to t = T
        where T = len(transmissions) and:
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
            if print_every and (t % print_every == 0):
                print(f"t = {t} / {T}")
            reset_probas(t, probas, observations)
            infection_probas = get_infection_probas_mean_field(
                probas[t], transmissions[t]
            )
            probas[t+1] = propagate(
                probas[t], infection_probas, recover_probas
            )
        self.probas = probas
        self.states = probas.argmax(axis=2)

# alias for backward compatibility
InferenceModel = MeanField

class DynamicMessagePassing(BaseInference):

    def time_evolution(self, recover_probas, transmissions, observations=[], print_every=10):
        """
        Run the probability evolution from t = 0 to t = T - 1
        where T = len(transmissions) and:
        - recover_probas[i] = mu_i time-independent
        - transmissions[t] = csr sparse matrix of i, j, lambda_ij(t)
        - observations = list of dict(i=i, s=s, t=t) observations at t_obs=t
        If s=I, the observation must also give t_I the infection time
        - probas[t, i, s] = P_s^i(t)
        """
        # initialize messages
        kappa, P_bar, phi = initial_messages(self.initial_probas)
        full_contacts = get_full_contacts(transmissions)
        P_bar = full_contacts.multiply(P_bar)
        phi = full_contacts.multiply(phi)
        records = [] # DEBUG
        # initialize probas
        T = len(transmissions)
        probas = np.zeros((T, self.N, 3))
        probas[0] = self.initial_probas
        # iterate over time steps
        for t in range(T):
            if print_every and (t % print_every == 0):
                print(f"t = {t} / {T}")
            reset_probas(t, probas, observations)
            if (t >= T-1):
                break
            # update
            contacts = (transmissions[t] != 0)
            kappa_next, P_bar_next, phi_next = update_messages(
                kappa, P_bar, phi, transmissions[t], recover_probas, full_contacts
            )
            infection_probas = get_infection_probas_dmp(
                kappa, contacts, kappa_next
            )
            probas[t+1] = propagate(
                probas[t], infection_probas, recover_probas
            )
            # DEBUG : record info
            rho = get_infection_messages(kappa, contacts, kappa_next, full_contacts)
            delta_P_bar = rho - rho.multiply(P_bar)
            records.append(infos_csr(t, "transmissions", transmissions[t]))
            records.append(infos_csr(t, "contacts", contacts))
            records.append(infos_array(t, "probas", probas[t]))
            records.append(infos_csr(t, "kappa", kappa))
            records.append(infos_csr(t, "P_bar", P_bar))
            records.append(infos_csr(t, "phi", phi))
            records.append(infos_csr(t, "rho", rho))
            records.append(infos_csr(t, "delta_P_bar", delta_P_bar))
            records.append(infos_csr(t, "kappa_next", kappa_next))
            records.append(infos_csr(t, "P_bar_next", P_bar_next))
            records.append(infos_csr(t, "phi_next", phi_next))
            records.append(infos_array(t, "infection_probas", infection_probas))
            # next iteration
            kappa, P_bar, phi = kappa_next, P_bar_next, phi_next
        self.records = pd.DataFrame(records) # DEBUG
        self.probas = probas
        self.states = probas.argmax(axis=2)
