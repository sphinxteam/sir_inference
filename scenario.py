import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sir_model import (
    get_infection_probas, propagate, STATES,
    infected_individuals, random_individuals, symptomatic_individuals
)
from ranking import RANKINGS, csr_to_list


def inactivate_transmission(transmission, quarantined):
    not_quarantined = 1 - quarantined
    # lambda_ij = 0 for i quarantined
    transmission_new = transmission.T.multiply(not_quarantined).T
    # lambda_ij = 0 for j quarantined
    transmission_new = transmission_new.multiply(not_quarantined)
    return transmission_new


def get_detected_by(observations, source):
    df = pd.DataFrame(observations)
    df = df[df.source==source].copy()
    df["detected"] = 1*(df["s"] == 1)
    df["tested"] = 1
    grouped = df.groupby("t_test")[["detected", "tested"]].sum().sort_index()
    grouped["total_detected"] = grouped["detected"].cumsum()
    grouped["total_tested"] = grouped["tested"].cumsum()
    return grouped


def get_counts(states, quarantine):
    counts = {
        s: (states == idx).sum(axis=1)
        for idx, s in enumerate(STATES)
    }
    counts["q"] = quarantine.sum(axis=1)
    counts = pd.DataFrame(counts)
    counts.index.name = "t"
    return counts


def get_status(states, quarantine):
    t_max = states.shape[0]
    # states wide -> long
    states = pd.DataFrame(states)
    states["t"] = range(t_max)
    states = states.melt(id_vars="t", var_name="i", value_name="s")
    # quarantine wide -> long
    quarantine = pd.DataFrame(quarantine)
    quarantine["t"] = range(t_max)
    quarantine = quarantine.melt(id_vars="t", var_name="i", value_name="q")
    # merge
    status = pd.merge(states, quarantine, on=["t", "i"], how="inner")
    return status


def get_obs_counts(observations):
    sources = ["ranking", "infected", "random", "symptomatic"]
    t_max = observations["t_test"].max()
    obs_count = observations.groupby(["source", "t_test", "s"]).size()
    full_index = pd.MultiIndex.from_product(
        [sources, range(t_max), range(3)], names=["source", "t_test", "s"]
    )
    obs_count = obs_count[full_index].fillna(0)
    keys = {idx: u for idx, u in enumerate(STATES)}
    obs_count = {
        source: obs_count.loc[source].unstack("s").rename(columns=keys)
        for source in sources
    }
    return obs_count


class Scenario():

    def __init__(self, model, seed,
                 ranking_options={},
                 observation_options={},
                 intervention_options={}):
        self.seed = seed
        self.N = model.N
        self.x_pos = model.x_pos
        self.y_pos = model.y_pos
        self.initial_states = model.initial_states
        self.recover_probas = model.recover_probas
        self.base_transmissions = model.transmissions
        self.ranking_options = ranking_options
        if ranking_options:
            self.ranking = RANKINGS[ranking_options["name"]]
        self.observation_options = observation_options
        self.intervention_options = intervention_options

    def update_states(self, t):
        "Assumes states[t-1] and transmissions[t-1] computed"
        if t == 0:
            self.states[t] = self.initial_states
            return
        infection_probas = get_infection_probas(
            self.states[t-1], self.transmissions[t-1]
        )
        self.states[t] = propagate(
            self.states[t-1], infection_probas, self.recover_probas
        )

    def update_scores(self, t):
        "Assumes observations[t-1] and transmissions[t-1] computed"
        self.scores[t] = self.ranking(
            t, self, self.observations, self.ranking_options
        )
        self.scores[t]["t"] = t

    def update_observations(self, t):
        "Assumes states[t] and scores[t] computed."
        # ranking
        n_obs = self.observation_options.get("n_ranking", 0)
        if n_obs and self.ranking_options:
            # list of people to test
            ranked = list(self.scores[t]["i"])
            already_detected = set(
                obs["i"] for obs in self.observations if obs["s"] == 1
            )
            selected = [i for i in ranked if i not in already_detected]
            selected = selected[:n_obs]
            self.observations += [
                dict(i=i, s=self.states[t, i], t_test=t, source="ranking")
                for i in selected
            ]
        # random
        n_obs = self.observation_options.get("n_random", 0)
        selected = random_individuals(self.N, n_obs)
        self.observations += [
            dict(i=i, s=self.states[t, i], t_test=t, source="random")
            for i in selected
        ]
        # infected
        n_obs = self.observation_options.get("n_infected", 0)
        selected = infected_individuals(self.states[t], n_obs)
        self.observations += [
            dict(i=i, s=self.states[t, i], t_test=t, source="infected")
            for i in selected
        ]
        # symptomatic
        p = self.observation_options.get("p_symptomatic", 0)
        if p:
            tau = self.observation_options["tau"]
            selected = symptomatic_individuals(self.states, t, tau, p)
            self.observations += [
                dict(i=i, s=self.states[t, i], t_test=t, source="symptomatic")
                for i in selected
            ]

    def update_quarantine(self, t):
        "Assumes observations[t] computed."
        detected = [
            obs["i"] for obs in self.observations
            if obs["t_test"] == t and obs["s"] == 1
        ]
        Q = self.intervention_options["quarantine_time"]
        # detected i in quarantine from t to t + Q
        t_end = min(t+Q, self.t_max)
        self.quarantine[t:t_end, detected] = 1

    def update_transmissions(self, t):
        "Assumes quarantine[t] computed."
        # inactivate the transmissions at t
        self.transmissions[t] = inactivate_transmission(
            self.transmissions[t], self.quarantine[t]
        )

    def save(self, filename):
        os.mkdir(filename)
        print(f"Saving to {filename}")
        self.status.to_csv(
            os.path.join(filename, "status.csv"), index=False
        )
        if self.observation_options:
            self.observations.to_csv(
                os.path.join(filename, "observations.csv"), index=False
            )
        if self.ranking_options:
            self.scores.to_csv(
                os.path.join(filename, "scores.csv"), index=False
            )
        print("Saving transmissions...")
        df_transmissions = pd.DataFrame(
            dict(t=t, i=i, j=j, lamb=lamb)
            for t, A in enumerate(self.transmissions)
            for i, j, lamb in csr_to_list(A)
        )
        df_transmissions.to_csv(
            os.path.join(filename, "transmissions.csv"), index=False
        )
        params = dict(
            N=self.N, seed=self.seed, t_max=self.t_max,
            ranking_options=self.ranking_options,
            observation_options=self.observation_options,
            intervention_options=self.intervention_options
        )
        fname = os.path.join(filename, "params.json")
        json.dump(
            params, open(fname, "w"), indent=4, separators=(',', ': ')
        )

    def update(self, t):
        self.update_states(t)
        if self.ranking_options:
            self.update_scores(t)
        if self.observation_options:
            self.update_observations(t)
        if self.intervention_options:
            self.update_quarantine(t)
            self.update_transmissions(t)

    def run(self, t_max, print_every=0):
        if print_every:
            print(f"Using seed={self.seed}")
        np.random.seed(self.seed)
        self.t_max = t_max
        # initialize
        self.states = np.zeros((t_max, self.N), dtype=int)
        self.quarantine = np.zeros((t_max, self.N), dtype=int)
        self.scores = [None for _ in range(t_max)]
        self.transmissions = self.base_transmissions.copy()
        self.observations = []
        # iterate
        for t in range(t_max):
            if print_every and (t % print_every == 0):
                print(f"t = {t} / {t_max}")
            self.update(t)
        # store as dataframes
        self.status = get_status(self.states, self.quarantine)
        self.counts = get_counts(self.states, self.quarantine)
        if self.ranking_options:
            self.scores = pd.concat(self.scores, ignore_index=True, sort=False)
        if self.observation_options:
            self.observations = pd.DataFrame(self.observations)
            self.observations = self.observations[["source", "t_test", "i", "s"]]
            self.obs_counts = get_obs_counts(self.observations)

    def plot(self, t):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for idx, state in enumerate(STATES):
            ind, = np.where(self.states[t] == idx)
            ax.scatter(self.x_pos[ind], self.y_pos[ind], label=state)
        observed = self.observations.query(f"t_test=={t}")["i"].values
        ax.scatter(self.x_pos[observed], self.y_pos[observed],
                   facecolors='none', edgecolors='k')
        quarantined, = np.where(self.quarantine[t])
        ax.scatter(self.x_pos[quarantined], self.y_pos[quarantined],
                   facecolors='none', edgecolors='r')
        ax.set(title="t = %d" % t)
        ax.legend()

    def compare_scores_status(self, t, exclude=True):
        if not self.ranking_options:
            raise ValueError("You need ranking options")
        status = self.status.query(f"t=={t}").drop(columns="t")
        scores = self.scores.query(f"t=={t}").drop(columns="t")
        merged = pd.merge(status, scores, on="i", how="inner")
        merged["infected"] = (merged["s"] == 1)
        assert merged.shape[0] == self.N
        if exclude:
            past_observations = self.observations.query(f"t_test < {t} & s==1")
            already_detected = past_observations["i"].unique()
            merged = merged[~merged["i"].isin(already_detected)]
            assert merged.shape[0] == self.N - len(already_detected)
        return merged

    def roc_curve(self, t, exclude=True):
        "If exclude = True we exclude already detected from the roc curve"
        merged = self.compare_scores_status(t, exclude)
        y_true = merged["infected"]
        y_score = merged["score"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        return fpr, tpr, auc

    def detected_curve(self, t, exclude=True):
        "If exclude = True we exclude already detected from the detected curve"
        merged = self.compare_scores_status(t, exclude)
        merged = merged.sort_values(by="rank")
        detected = merged["infected"].cumsum().values
        tested = 1 + np.arange(len(detected))
        return tested, detected

    def detected_by(self, source):
        return get_detected_by(self.observations, source)
