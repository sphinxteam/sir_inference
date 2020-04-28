import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from ipywidgets import interact, IntSlider

# sir_inference imports
from .inference_model import MeanField, DynamicMessagePassing
from .sir_model import (
    EpidemicModel, ProximityModel, 
    patient_zeros_states, frequency, indicator,
    random_individuals, infected_individuals,
)


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


def proximity_model():
    print("Using ProximityModel")
    np.random.seed(42)
    N = 2000 #number of individuals
    N_patient_zero = 10 #number of patients 0
    scale = 1.0 #scale of the graphs
    mu = 0.01 # mu -> recover
    lamb = 0.02 # lamb -> transmission
    initial_states = patient_zeros_states(N, N_patient_zero)
    model = ProximityModel(N, scale, mu, lamb, initial_states)
    print("expected number of contacts %.1f" % model.n_contacts)
    model.run(T=50, print_every=100)
    return model


def ferretti_model( N_patient_zero = 10, mu = 1/15, lamb = 0.02):
    print("Using Ferretti transmissions")
    N = 10000
    N_patient_zero = 10
    transmissions = read_ferretti_data("all_interaction_10000.csv", lamb=lamb)
    initial_states = patient_zeros_states(N, N_patient_zero)
    # random x_pos, y_pos
    x_pos = np.random.rand(N)
    y_pos = np.random.rand(N)
    model = EpidemicModel(initial_states=initial_states, x_pos=x_pos, y_pos=y_pos)
    mu = 1/15
    recover_probas = mu*np.ones(N)
    model.time_evolution(recover_probas, transmissions, print_every=100)
    return model

########################################### Utils ###########################################
## Ranking ##

def csr_to_list(x):
    x_coo = x.tocoo()
    return zip(x_coo.row, x_coo.col, x_coo.data)

# ranking functions
def ranking_inference(t, model, observations, params):
    """Inference starting from t_start.
    
    Run Mean Field from t_start to t, starting from all susceptible and 
    resetting the probas according to observations. The time of infection is given
    
    params["t_start"] : t_start
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)
    """
    t_start = params["t_start"]
    tau = params["tau"]
    algo = MeanField  if params["algo"]=="MF" else DynamicMessagePassing
    if params["init"]=="all_S":
        initial_probas = indicator(np.zeros(model.N))
    else:
        initial_probas = frequency(model.states[t_start])
    infer = algo(initial_probas, model.x_pos, model.y_pos)
    # shift by t_start
    for obs in observations:
        obs["t"] = obs["t_test"] - t_start
        obs["t_I"] = obs["t"] - tau
    infer.time_evolution(
        model.recover_probas, model.transmissions[t_start:t+1], observations,
        print_every=0
    )
    probas = pd.DataFrame(
        infer.probas[t-t_start,:,:], 
        columns=["p_S","p_I","p_R"]
    )
    probas["i"] = range(model.N)
    probas = probas.sort_values(by="p_I", ascending=False)
    ranked = list(probas["i"])
    return ranked

def ranking_inference_backtrack(t, model, observations, params):
    """Mean Field starting from t - delta.
    
    Run Mean Field from t - delta to t, starting from all susceptible and 
    resetting the probas according to observations.
    
    params["delta"] : delta
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)
    """
    t_start = t - params["delta"]
    tau = params["tau"]
    algo = MeanField  if params["algo"]=="MF" else DynamicMessagePassing
    if params["init"]=="all_S":
        initial_probas = indicator(np.zeros(model.N))
    else:
        initial_probas = frequency(model.states[t_start])
    infer = algo(initial_probas, model.x_pos, model.y_pos)
    # shift by t_start
    for obs in observations:
        obs["t"] = obs["t_test"] - t_start
        obs["t_I"] = obs["t"] - tau
    infer.time_evolution(
        model.recover_probas, model.transmissions[t_start:t+1], observations,
        print_every=0
    )
    probas = pd.DataFrame(
        infer.probas[t-t_start,:,:], 
        columns=["p_S","p_I","p_R"]
    )
    probas["i"] = range(model.N)
    probas = probas.sort_values(by="p_I", ascending=False)
    ranked = list(probas["i"])
    return ranked


def ranking_random(t, model, observations, params):
    """Returns a random ranking."""
    ranked = np.random.permutation(model.N)
    return ranked


def ranking_tracing(t, model, observations, params):
    """Naive contact tracing. 
    
    Search for all individuals that have been in contact during [t-delta, t] 
    with the individuals last tested positive (observations s=I at t_test=t-1).
    
    params["delta"] = delta
    """
    delta = params["delta"]
    # last_tested : observations s=I at t_test=t-1
    last_tested = set(
        obs["i"] for obs in observations 
        if obs["s"]==1 and obs["t_test"]==t-1
    )
    # contacts with last_tested people during [t - delta, t]
    contacts = pd.DataFrame(
        dict(i=i, j=j, t=t_contact) 
        for t_contact in range(t - delta, t)
        for i, j, _ in csr_to_list(model.transmissions[t_contact])
        if j in last_tested
    )
    no_contact = contacts.shape[0]==0 
    if no_contact:
        return np.random.permutation(model.N)
    # number of encounters for all i 
    counts = contacts.groupby("i").size()
    encounters = pd.DataFrame({"i":range(model.N)})
    # if i has no encounters with last_tested people, count is zero
    encounters["count"] = encounters["i"].map(counts).fillna(0)
    encounters = encounters.sort_values(by="count", ascending=False)
    ranked = list(encounters["i"])
    return ranked

RANKINGS = {
    "inference":ranking_inference,
    "backtrack":ranking_inference_backtrack,
    "tracing":ranking_tracing,
    "random":ranking_random
}

## Observations ##

def run_observations(initial_obs, model, params):
    # initial observations
    observations = initial_obs.copy()
    # parse args
    use_ranking = params["n_test"]["ranking"] > 0 
    if use_ranking:
        ranking_name = params["ranking"]
        ranking = RANKINGS[ranking_name]
        print(f"Using {ranking.__name__} to rank")
    # iterate
    for t in range(params["t_start"], params["t_final"]):
        # ranking
        if use_ranking:
            # list of people to test
            ranked = ranking(t, model, observations, params)
            already_detected = set(
                obs["i"] for obs in observations if obs["s"]==1
            )
            selected = [i for i in ranked if i not in already_detected]
            selected = selected[:params["n_test"]["ranking"]]
            observations += [
                dict(i=i, s=model.states[t, i], t_test=t, source="ranking") 
                for i in selected
            ]
        # random
        selected = random_individuals(model, n_obs=params["n_test"]["random"])
        observations += [
            dict(i=i, s=model.states[t, i], t_test=t, source="random") 
            for i in selected
        ]
        # infected
        selected = infected_individuals(model, t, n_obs=params["n_test"]["infected"])
        observations += [
            dict(i=i, s=model.states[t, i], t_test=t, source="infected") 
            for i in selected
        ]
    # format output
    keys = ["i","s","t_test","source"]
    observations = [
        {key:obs[key] for key in keys} for obs in observations
    ]
    return observations


def ranking_observations(t, model, past_observations, params):
    ranking_name = params["ranking"]
    ranking = RANKINGS[ranking_name]
    # list of people to test
    ranked = ranking(t, model, past_observations, params)
    already_detected = set(
        obs["i"] for obs in past_observations if obs["s"]==1
    )
    selected = [i for i in ranked if i not in already_detected]
    observations = [
        dict(i=i, s=model.states[t, i], t_test=t, source="ranking", rank=rank) 
        for rank, i in enumerate(selected)
    ]
    df = pd.DataFrame(observations)
    df["detected"] = df["s"]==1
    df = df.sort_values(by="rank")
    df["total_detected"] = df["detected"].cumsum() 
    return df


def get_detected(observations):
    df = pd.DataFrame(observations)
    df = df.query("source=='ranking'").copy()
    df["detected"] = df["s"]==1
    df["tested"] = 1
    grouped = df.groupby("t_test")[["detected","tested"]].sum()
    grouped = grouped.reset_index().sort_values(by="t_test")
    grouped["total_detected"] = grouped["detected"].cumsum()
    grouped["total_tested"] = grouped["tested"].cumsum()
    return grouped
