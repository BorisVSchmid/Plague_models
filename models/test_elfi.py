import elfi
import numpy as np
from os.path import dirname, abspath
import os.path

def sir(beta, gamma, batch_size=1, random_state=None):
    population = 10000
    suseptibles = np.zeros(50)
    suseptibles[0] = population
    infected = np.zeros(50)
    infected[0] = 20.0
    recovered = np.zeros(50)
    for i in range(1, 49):
        number_of_humans = infected[i - 1] + suseptibles[i - 1] + recovered[i - 1]
        n_infected = beta * ((suseptibles[i - 1]*infected[i-1])/number_of_humans)
        n_recovered = gamma * infected[i - 1]
        # new states
        suseptibles[i] = suseptibles[i - 1] - n_infected
        infected[i] = infected[i - 1] + n_infected - n_recovered
        recovered[i] = n_recovered
    return recovered


def sir_elfi(beta, gamma, batch_size=1, random_state=None):
    population = 10000
    ini_infected = 20.0
    suseptibles = np.asarray([np.zeros(batch_size) for _ in range(50)])
    suseptibles[0][:] = population
    infected = np.asarray([np.zeros(batch_size) for _ in range(50)])
    infected[0][:] = ini_infected
    recovered = np.asarray([np.zeros(batch_size) for _ in range(50)])
    for i in range(1, 49):
        number_of_humans = np.add(np.add(infected[i - 1], suseptibles[i - 1]), recovered[i - 1])
        n_infected = np.multiply(beta, np.divide(np.multiply(suseptibles[i - 1], infected[i - 1]), number_of_humans))
        n_recovered = np.multiply(gamma, infected[i - 1])
        # new states
        suseptibles[:][i] = np.subtract(suseptibles[i - 1], n_infected)
        infected[:][i] = np.subtract(np.add(infected[i - 1], n_infected), n_recovered)
        recovered[:][i] = n_recovered
    return recovered


def test(x, observed, batch_size=10000):
    return np.mean(np.subtract(x, np.asarray([np.asarray([b for _ in range(batch_size)]) for b in observed])))


if __name__ == "__main__":
    obs = sir(0.5, 0.4)
    beta = elfi.Prior('uniform', 0.2, 1.0)
    gamma = elfi.Prior('uniform', 0.1, 0.7)
    Y = elfi.Simulator(sir_elfi, beta, gamma, observed=obs)
    a = elfi.Summary(test, Y, obs)
    rej = elfi.Rejection(a, batch_size=10000)
    N = 1000
    result = rej.sample(N, quantile=0.01)
