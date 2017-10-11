import pymc as pm
import numpy as np
import pandas as pd
from datetime import date


md = []
with open("sim_md.csv", mode='r') as file:
    [md.append(float(a)) for a in file.read().split(', ')]

__all__ = ['md', 'beta', 's_h', 'gamma_h', 'p_recovery_h', 'fraction', 'sigma', 'i_r0', 's_r0', 'gamma_r',
           'p_recovery_ur', 'rep_rate_r', 'rep_rate_ur', 'inh_res', 'd_rate_ui', 'd_rate', 'g_rate', 'c_cap',
           'fph0', 'searching', 'd_', 'mortality', 'mortality_sim']

# years = mdates.YearLocator()  # every year
# months = mdates.MonthLocator()  # every month
# yearsFmt = mdates.DateFormatter('%Y')
years_list = pd.date_range(date(1990, 1, 1), date(1990, 12, 31)).tolist()

# -- Params
temp = [[31.1, 27.1, 23.6, 82], [30.8, 27.2, 23.8, 83], [32.5, 27.5, 23.6, 81], [32.5, 27.4, 22.9, 73],
        [32.0, 26.1, 20.6, 67], [31.0, 24.6, 18.6, 64], [30.8, 24.2, 18.0, 62], [31.4, 24.6, 18.4, 60],
        [32.1, 25.4, 19.6, 63], [32.5, 26.8, 22.0, 66], [32.2, 27.7, 23.5, 72], [31.3, 27.4, 23.7, 80]]
decade = {}
for time in years_list:
    time = time.strftime("%Y-%m-%d")
    time_s = time.split("-")
    decade[time] = temp[int(time_s[1]) - 1]
decade_list = [x for x in decade.keys()]
t = [x for x in range(0, len(decade))]

# - Human
beta = pm.Uniform('beta', lower=1e-3, upper=.2, value=0.1)
s_h = 25000.
gamma_h = 0.1
p_recovery_h = .4

# - rat
fraction = pm.Uniform('fraction', lower=1e-3, upper=1., value=.5)
sigma = pm.Uniform('sigma', lower=1e-3, upper=1., value=.5)
i_r0 = pm.Uniform('i_r0', lower=1., upper=17., value=9.)
# 0.08
s_r0 = s_h * fraction
# rec_r[0] = s_h * fraction
res_r0 = 0.
gamma_r = 0.2
# .1 \/
p_recovery_ur = .1
rep_rate_r = .4 * (1 - 0.234)
rep_rate_ur = .4
inh_res = 0.975
d_rate_ui = 1 / (365 * 1)

# - flea
d_rate = 0.2
# 0.2
g_rate = .0084
c_cap = 6.
fph0 = c_cap
searching = 3. / (s_r0 + res_r0)

@pm.deterministic
def plague_model(s_r0=s_r0, res_r0=res_r0, i_r0=i_r0, fph0=fph0, beta=beta, fraction=fraction, sigma=sigma):
    # -- Params
    # - Human
    i_h = np.zeros_like(t, dtype=float)
    r_h = np.zeros_like(t, dtype=float)
    d_h = np.zeros_like(t, dtype=float)
    d_h[0] = 0.0000001

    # - rat
    s_r = np.zeros_like(t, dtype=float)
    i_r = np.zeros_like(t, dtype=float)
    res_r = np.zeros_like(t, dtype=float)
    d_r = np.zeros_like(t, dtype=float)
    i_r[0] = i_r0
    s_r[0] = s_r0
    res_r[0] = res_r0

    # - flea
    i_f = np.zeros_like(t, dtype=float)
    fph = np.zeros_like(t, dtype=float)
    fph[0] = fph0

    # -- Simulate
    for i, v in enumerate(decade_list[1:], 1):
        temp_fac = ((decade[v][1] - 18)/((14/3)*4)) + (1 / 4)
        # + rec_r[i - 1]
        N_r = s_r[i - 1] + i_r[i - 1] + res_r[i - 1]
        # - Fleas
        if i == 1:
            infected_rat_deaths = d_h[0]
            # avg number of fleas per rat at carrying capacity
            c_cap = fph[0]
        if fph[i - 1] / c_cap < 1.:
            flea_growth = (g_rate * temp_fac) * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        elif fph[i - 1] / c_cap > 1.:
            flea_growth = -(g_rate * temp_fac) * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        else:
            flea_growth = 0.

        new_infectious = infected_rat_deaths * fph[i - 1]
        starvation_deaths = d_rate * i_f[i - 1]
        # number of fleas that find a human
        force_to_humans = i_f[i - 1] * np.exp(float(-searching * N_r))
        # number of fleas that find a rat
        force_to_rats = i_f[i - 1] - force_to_humans
        fph[i] = fph[i - 1] + flea_growth
        i_f[i] = i_f[i - 1] + new_infectious - starvation_deaths

        # - Rats
        new_infected_rats = sigma * s_r[i - 1] * force_to_rats / N_r
        new_infected_rats = 0 if new_infected_rats < 0 else new_infected_rats
        new_removed_rats = gamma_r * i_r[i - 1]
        new_recovered_rats = p_recovery_ur * new_removed_rats
        new_dead_rats = new_removed_rats - new_recovered_rats
        infected_rat_deaths = new_dead_rats

        # born rats
        pressure = 0 if (N_r / (s_h * fraction)) < 0 else (N_r / (s_h * fraction))
        resistant_born_rats = (rep_rate_r * res_r[i - 1] * (inh_res - pressure))
        unresistant_from_resistant = (rep_rate_ur * res_r[i - 1] * (1. - inh_res))
        unresistant_born_rats = (rep_rate_ur * (res_r[i - 1] + s_r[i - 1]) * (1. - pressure))
        born_rats = unresistant_born_rats + unresistant_from_resistant

        # natural deaths
        natural_death_unresistant = (s_r[i - 1] * d_rate_ui)
        natural_death_resistant = (res_r[i - 1] * d_rate_ui)
        # rec

        # time step values
        s_r[i] = s_r[i - 1] + born_rats - new_infected_rats - natural_death_unresistant
        i_r[i] = i_r[i - 1] + new_infected_rats - new_removed_rats
        # rec_r[i] = rec_r[i - 1] + new_recovered_rats - natural_death_recovered
        res_r[i] = res_r[i - 1] + new_recovered_rats + resistant_born_rats - natural_death_resistant
        d_r[i] = new_dead_rats + natural_death_unresistant + natural_death_resistant

        # - Humans
        N_h = s_h + i_h[i - 1] + r_h[i - 1]
        new_infected_humans = beta * s_h * force_to_humans / N_h
        new_removed_humans = gamma_h * i_h[i - 1]
        new_recovered_humans = p_recovery_h * new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans

        # time step values
        i_h[i] = i_h[i - 1] + new_infected_humans - new_removed_humans
        r_h[i] = r_h[i - 1] + new_recovered_humans
        d_h[i] = 0.0001 + new_dead_humans
    return i_h, r_h, d_h, s_r, i_r, res_r, d_r, i_f, fph


d_ = pm.Lambda('d_', lambda plague_model=plague_model: plague_model[2])
md = np.asarray(md, dtype=float)

#Likelihood
mortality = pm.Poisson('mortality', mu=d_, value=md, observed=True)
mortality_sim = pm.Poisson('mortality_sim', mu=d_)
