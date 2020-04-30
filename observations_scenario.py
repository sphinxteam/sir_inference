"""Deprecated use the Scenario class instead."""

import numpy as np
import pandas as pd
from sir_model import random_individuals, infected_individuals
from ranking import RANKINGS


def run_observations(initial_obs, model, params):
    # initial observations
    observations = initial_obs.copy()
    # parse args
    use_ranking = params["n_test"]["ranking"] > 0
    if use_ranking:
        ranking_name = params["ranking"]
        ranking = RANKINGS[ranking_name]
    # iterate
    for t in range(params["t_start"], params["t_final"]):
        # ranking
        if use_ranking:
            # list of people to test
            scores = ranking(t, model, observations, params)
            ranked = scores["i"].values
            already_detected = set(
                obs["i"] for obs in observations if obs["s"] == 1
            )
            selected = [i for i in ranked if i not in already_detected]
            selected = selected[:params["n_test"]["ranking"]]
            observations += [
                dict(i=i, s=model.states[t, i], t_test=t, source="ranking")
                for i in selected
            ]
        # random
        n_obs = params["n_test"]["random"]
        selected = random_individuals(model.N, n_obs)
        observations += [
            dict(i=i, s=model.states[t, i], t_test=t, source="random")
            for i in selected
        ]
        # infected
        n_obs=params["n_test"]["infected"]
        selected = infected_individuals(model.states[t], n_obs)
        observations += [
            dict(i=i, s=model.states[t, i], t_test=t, source="infected")
            for i in selected
        ]
    # format output
    keys = ["i", "s", "t_test", "source"]
    observations = [
        {key: obs[key] for key in keys} for obs in observations
    ]
    return observations


def ranking_observations(t, model, past_observations, params):
    ranking_name = params["ranking"]
    ranking = RANKINGS[ranking_name]
    # merge scores wih actual states
    scores = ranking(t, model, past_observations, params)
    status = pd.DataFrame({
        "i":range(model.N), "s":model.states[t, :]
    })
    scores = pd.merge(scores, status, on="i", how="inner")
    assert scores.shape[0]==model.N
    # exclude already detected
    already_detected = set(
        obs["i"] for obs in past_observations if obs["s"] == 1
    )
    scores = scores[~scores["i"].isin(already_detected)]
    # rerank
    scores = scores.sort_values(by="rank")
    scores["infected"] = scores["s"] == 1
    scores["tested"] = 1 + np.arange(scores.shape[0])
    scores["detected"] = scores["infected"].cumsum()
    return scores
