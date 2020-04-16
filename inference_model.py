import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt

STATES = ["S", "I", "R"]


def infos_csr(t, name, array):
    return dict(
        t=t, name=name, shape=array.shape, nnz=array.nnz,
        nan=(array == np.nan).sum(), min=array.min(), max=array.max()
    )


def infos_array(t, name, array):
    return dict(
        t=t, name=name, shape=array.shape, nnz=(array != 0).sum(),
        nan=(array == np.nan).sum(), min=array.min(), max=array.max()
    )


def zero_csr(N):
    i = j = data = []
    return csr_matrix((data, (i, j)), shape=(N, N))


def update_dmp(history, kappa, P_bar, phi,
               P_bar_vec, phi_vec, probas, transmissions, recover_probas):
    """
    Parameters
    ----------
    - history = csr sparse matrix of i, j in contact < t
    - kappa = csr sparse matrix of i, j, kappa_ij(t)
    - P_bar = csr sparse matrix of i, j, P_bar_ij(t)
    - phi = csr sparse matrix of i, j, phi_ij(t)
    - P_bar_vec[j] = P_bar^j(t) = 1 - P_S^j(t)
    - phi_vec[j] = phi^j(t)
    - probas[j,s] = P_s^j(t)

    - transmissions = csr sparse matrix of i, j, lambda_ij(t)
    - recover_probas[j] = mu_j

    Returns
    -------
    - history_next = csr sparse matrix of i, j in contact < t+1
    - kappa_next = csr sparse matrix of i, j, kappa_ij(t+1)
    - P_bar_next = csr sparse matrix of i, j, P_bar_ij(t+1)
    - phi_next = csr sparse matrix of i, j, phi_ij(t+1)
    - P_bar_vec_next[j] = P_bar^j(t+1) = 1 - P_S^j(t+1)
    - phi_vec_next[j] = phi^j(t+1)
    - probas_next[j,s] = P_s^j(t+1)
    """
    # set P_bar_vec
    P_bar_vec = 1 - probas[:, 0]
    # deal with contacts
    contacts = (transmissions != 0)                             # ij = t
    history_next = contacts.maximum(history)                    # ij = t or < t
    new_contacts = contacts - contacts.minimum(history)         # ij = t &! < t
    # store new contacts
    P_bar_all = P_bar + new_contacts.multiply(P_bar_vec)        # ij = t or < t
    phi_all = phi + new_contacts.multiply(phi_vec)              # ij = t or < t
    # update kappa
    kappa_next = kappa + transmissions.multiply(phi_all)        # ij = t or < t
    # infection probas
    L = kappa.multiply(-1).log1p().multiply(contacts)           # ij = t
    L_next = kappa_next.multiply(-1).log1p().multiply(contacts) # ij = t
    dL = L_next - L                                             # ij = t
    ones = np.ones(kappa.shape[0])
    da = dL.dot(ones)
    rho_vec = -np.expm1(da)
    # infection messages
    dA = history_next.multiply(da) - dL                         # ij = t or < t
    rho = dA.expm1().multiply(-1)                               # ij = t or < t
    # update P_bar and phi
    delta_P_bar = rho - rho.multiply(P_bar_all)                 # ij = t or < t
    P_bar_next = P_bar_all + delta_P_bar
    phi_next = (
        (phi_all - phi_all.multiply(transmissions)).multiply(1 - recover_probas)
        + delta_P_bar
    )
    # update vectors
    delta_P_bar_vec = rho_vec*(1 - P_bar_vec)
    P_bar_vec_next = P_bar_vec + delta_P_bar_vec
    phi_vec_next = phi_vec*(1 - recover_probas) + delta_P_bar_vec
    # update probas
    probas_next = propagate(probas, rho_vec, recover_probas)
    # sanity check
    assert np.allclose(P_bar_vec, 1 - probas[:, 0])
    assert np.allclose(P_bar_vec_next, 1 - probas_next[:, 0])
    return (
        history_next,
        kappa_next, P_bar_next, phi_next,
        P_bar_vec_next, phi_vec_next, probas_next
    )


############### Mean field ##############################
def get_infection_probas_mean_field(probas, transmissions):
    """
    - probas[i,s] = P_s^i(t)
    - transmissions = csr sparse matrix of i, j, lambda_ij(t)
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
    - transmissions[t] = csr sparse matrix of i, j, lambda_ij(t)
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
        history = kappa = P_bar = phi = zero_csr(self.N)
        P_bar_vec = 1 - self.initial_probas[:, 0]
        phi_vec = self.initial_probas[:, 1]
        records = []  # DEBUG
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
            new = update_dmp(
                history, kappa, P_bar, phi, P_bar_vec, phi_vec, probas[t],
                transmissions[t], recover_probas
            )
            history, kappa, P_bar, phi, P_bar_vec, phi_vec, probas[t+1] = new
            # DEBUG : record info
            records.append(infos_csr(t, "transmissions", transmissions[t]))
            records.append(infos_csr(t, "history", history))
            records.append(infos_csr(t, "kappa", kappa))
            records.append(infos_csr(t, "P_bar", P_bar))
            records.append(infos_csr(t, "phi", phi))
            records.append(infos_array(t, "P_bar_vec", P_bar_vec))
            records.append(infos_array(t, "phi_vec", phi_vec))
            records.append(infos_array(t, "probas", probas[t]))
        self.records = pd.DataFrame(records)  # DEBUG
        self.probas = probas
        self.states = probas.argmax(axis=2)
