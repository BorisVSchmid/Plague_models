import pymc3 as pm
import numpy as np


def run():
    # a list of observed deaths
    deaths = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
              2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9,
              9, 10, 11, 11, 0]
    with pm.Model() as model:
        gamma = pm.Uniform(lower=.3, upper=.5, name='gamma')
        beta = pm.Uniform(lower=1e-9, upper=1., name='beta')
        # set up a simple sir model
        s = -beta * s * i / (s + i)
        i = beta * s * i / (s + i) - (gamma * i)
        d = pm.Poisson('d', mu=gamma * i, observed=deaths)

    with model:
        pm.fit()
        trace = pm.sample()


def run2():
    # a list of observed deaths
    deaths = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
              2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9,
              9, 10, 11, 11, 0]
    with pm.Model() as model:
        gamma = pm.Uniform(lower=.3, upper=.5, name='gamma')
        beta = pm.Uniform(lower=1e-9, upper=1., name='beta')
        span = [x for x in range(0, len(deaths))]
        # set up a simple sir model
        s = np.zeros_like(span, dtype=object)
        i = np.zeros_like(span, dtype=object)
        d = np.zeros_like(span, dtype=object)
        for t in span[1:]:
            infected = beta * s[t - 1] * i[t - 1] / (s[t - 1] + i[t - 1])
            dead = (gamma * i[t - 1])
            s[t] = s[t - 1] - infected
            i[t] = i[t - 1] + infected - dead
            d[t] = dead
        obs = pm.Poisson('obs', mu=d, observed=deaths)

    with model:
        pm.fit()
        trace = pm.sample()


def run3():
    gamma = .4
    beta = .5
    span = 50
    # set up a simple sir model
    s = np.zeros(50)
    s[0] = 500.
    i = np.zeros(50)
    i[0] = .5
    d = np.zeros(50)
    d[0] = 0.
    for t in range(1, span - 1):
        infected = beta * s[t - 1] * i[t - 1] / (s[t - 1] + i[t - 1])
        dead = (gamma * i[t - 1])
        s[t] = s[t - 1] - infected
        i[t] = i[t - 1] + infected - dead
        d[t] = int(dead)
    print(d)
