import numpy as np
import pandas as pd
from inference_model import MeanField, DynamicMessagePassing
from sir_model import frequency, indicator


def csr_to_list(x):
    x_coo = x.tocoo()
    return zip(x_coo.row, x_coo.col, x_coo.data)


def ranking_inference(t, model, observations, params):
    """Inference starting from t_start.

    Run Mean Field from t_start to t, starting from all susceptible and
    resetting the probas according to observations.

    params["t_start"] : t_start
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe probas[["i","rank","score","p_I","p_R","p_S"]]
    """
    t_start = params["t_start"]
    tau = params["tau"]
    if (t < t_start):
        print(f"cannot do ranking_inference for t={t} < t_start={t_start}")
        return ranking_random(t, model, observations, params)
    algo = MeanField if params["algo"] == "MF" else DynamicMessagePassing
    if params["init"] == "all_S":
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
        infer.probas[t-t_start, :, :],
        columns=["p_S", "p_I", "p_R"]
    )
    probas["i"] = range(model.N)
    # some i will have the same probas
    # -> we add a random value to shuffle the ranking
    probas["rand"] = np.random.rand(model.N)
    probas = probas.sort_values(by=["p_I", "rand"], ascending=False)
    probas.reset_index(drop=True, inplace=True)
    probas["rank"] = range(model.N)
    probas["score"] = probas["p_I"]
    return probas


def ranking_backtrack(t, model, observations, params):
    """Mean Field starting from t - delta.

    Run Mean Field from t - delta to t, starting from all susceptible and
    resetting the probas according to observations.

    params["delta"] : delta
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe probas[["i","rank","score","p_I","p_R","p_S"]]
    """
    delta = params["delta"]
    tau = params["tau"]
    if (t < delta):
        print(f"cannot do ranking_backtrack for t={t} < delta={delta}")
        return ranking_random(t, model, observations, params)
    t_start = t - delta
    algo = MeanField if params["algo"] == "MF" else DynamicMessagePassing
    if params["init"] == "all_S":
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
        infer.probas[t-t_start, :, :], columns=["p_S", "p_I", "p_R"]
    )
    probas["i"] = range(model.N)
    # some i will have the same probas
    # -> we add a random value to shuffle the ranking
    probas["rand"] = np.random.rand(model.N)
    probas = probas.sort_values(by=["p_I", "rand"], ascending=False)
    probas.reset_index(drop=True, inplace=True)
    probas["rank"] = range(model.N)
    probas["score"] = probas["p_I"]
    return probas


def ranking_random(t, model, observations, params):
    """Random ranking.

    Returns: ranked dataframe df[["i","rank","score"]]
    """
    ranked = np.random.permutation(model.N)
    df = pd.DataFrame({
        "i":ranked, "rank":range(model.N), "score":np.linspace(1, 0, model.N)
    })
    return df


def ranking_tracing(t, model, observations, params):
    """Naive contact tracing.

    Search for all individuals that have been in contact during [t-tau, t]
    with the individuals last tested positive (observations s=I at t_test=t-1).

    params["tau"] = tau

    Returns: ranked dataframe encounters[["i","rank","score","count"]]
    """
    tau = params["tau"]
    if (t < tau):
        print(f"cannot do ranking_tracing for t={t} < tau={tau}")
        return ranking_random(t, model, observations, params)
    # last_tested : observations s=I at t_test=t-1
    last_tested = set(
        obs["i"] for obs in observations
        if obs["s"] == 1 and obs["t_test"] == t-1
    )
    # contacts with last_tested people during [t - tau, t]
    contacts = pd.DataFrame(
        dict(i=i, j=j, t=t_contact)
        for t_contact in range(t - tau, t)
        for i, j, _ in csr_to_list(model.transmissions[t_contact])
        if j in last_tested
    )
    encounters = pd.DataFrame({"i": range(model.N)})
    # no encounters -> count = 0
    if (contacts.shape[0] == 0):
        encounters["count"] = 0
    else:
        counts = contacts.groupby("i").size() # number of encounters for all i
        encounters["count"] = encounters["i"].map(counts).fillna(0)
    # many i will have the same count
    # -> we add a random value to shuffle the ranking
    encounters["rand"] = np.random.rand(model.N)
    encounters = encounters.sort_values(by=["count", "rand"], ascending=False)
    encounters.reset_index(drop=True, inplace=True)
    encounters["rank"] = range(model.N)
    encounters["score"] = encounters["count"]
    return encounters


def ranking_tracing_backtrack(t, model, observations, params):
    """Naive contact tracing + backtrack

    First rank according to contact tracing (past contact or not), then by
    the MF/DMP probas.

    params["delta"] : delta
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe df[["i","rank","score","count","p_I","p_R","p_S"]]
    """
    tau = params["tau"]
    if (t < tau):
        print(f"cannot do ranking_tracing_backtrack for t={t} < tau={tau}")
        return ranking_random(t, model, observations, params)
    delta = params["delta"]
    if (t < delta):
        print(f"cannot do ranking_tracing_backtrack for t={t} < delta={delta}")
        return ranking_random(t, model, observations, params)
    encounters = ranking_tracing(t, model, observations, params)
    encounters.drop(columns=["rank","score"], inplace=True)
    probas = ranking_backtrack(t, model, observations, params)
    probas.drop(columns=["rank","score"], inplace=True)
    df = pd.merge(encounters, probas, on=["i"], how="inner")
    df["past_contact"] = 1*(df["count"] > 0)
    df["score"] = df["past_contact"] + df["p_I"]
    # some i will have the same score
    # -> we add a random value to shuffle the ranking
    df["rand"] = np.random.rand(model.N)
    df = df.sort_values(by=["score", "rand"], ascending=False)
    df.reset_index(drop=True, inplace=True)
    df["rank"] = range(model.N)
    return df


RANKINGS = {
    "tracing_backtrack": ranking_tracing_backtrack,
    "inference": ranking_inference,
    "backtrack": ranking_backtrack,
    "tracing": ranking_tracing,
    "random": ranking_random
}
