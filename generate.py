from sir_model import FastProximityModel

def save_proximity_model(N, t_max, name, ext="csv.gz"):
    model = FastProximityModel(N=N, scale=1., mu=0.01, lamb=0.02)
    model.generate_transmissions(t_max, print_every=1)
    model.save_transmissions(f"{name}_transmissions.{ext}")
    model.save_positions(f"{name}_positions.{ext}")

if __name__=="__main__":
    save_proximity_model(N=100_000, t_max=100, name="prox_100K", ext="csv.gz")
