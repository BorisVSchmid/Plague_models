"""This model is the rat flea human with seasonality model made to be fitted against mortality data"""
import pymc as pm
import numpy as np
import pandas as pd
from datetime import date
from tools.TempReader import TempReader
from os.path import dirname, abspath
import os.path


__all__ = ['confirmed_cases', 'sigma', 'beta', 'gamma_h', 'p_recovery_h', 'phi', 'rho', 'gamma_r', 'p_recovery_ur', 'rep_rate_r',
           'rep_rate_ur', 'iota', 'theta', 'epsilon', 'd_rate_ui', 'd_rate', 'g_rate', 'c_cap', 'sim_data', 'mortality', 'mortalitysim']
dir = dirname(dirname(abspath(__file__)))
start_year = 1995
end_year = 1999
# confirmed_cases = np.array(list(map(float, open(os.path.join(dir, 'data', 'sim_md90.csv'), 'r').read().split(', '))), dtype=float)
confirmed_cases = [0, 0, 1, 0, 2, 0, 8, 12, 62, 16, 2, 14, 6, 5, 0, 0, 0, 0, 1, 5, 22, 39, 11, 8, 5, 6, 2, 1, 0, 0, 10, 38, 59, 74,
                   13, 6, 1, 1, 0, 0, 0, 0, 4, 17, 18, 29, 9, 8, 3, 3, 1, 0, 1, 0]
years_list = pd.date_range(date(start_year, 1, 1), date(end_year, 7, 1)).tolist()
# -- Params
data, temp_list = TempReader().cooked()
t = len(data)
# - human
sigma = pm.Uniform('sigma', 10000.0, 30000.0, value=20000)
beta = pm.Uniform('beta', 0.001, 1.0, value=0.01)
# .2
gamma_h = 0.1
p_recovery_h = .4
# - rat
# .08
phi = pm.Uniform('phi', 0.01, 1.0, value=.12)
# 0.08
rho = pm.Uniform('rho', 1e-3, .2, value=0.1)
# .2
gamma_r = 0.2
# .1
p_recovery_ur = .1
rep_rate_r = .4 * (1 - 0.234)
rep_rate_ur = .4
iota = pm.Uniform('iota', 0.5, 0.975, value=0.8)
theta = pm.Uniform('theta', 15, 25, value=18)
d_rate_ui = 1 / (365 * 1)
# - flea
d_rate = 0.2
# 0.2
g_rate = .0084
c_cap = 6.
# -- Simulate
@pm.deterministic
def plague_model(s_h=sigma, beta_h=beta, sus_frac=phi, beta_r=rho, inh_res=iota, i_r0=theta):
    #human
    i_h = np.zeros(t, dtype=float)
    i_n = np.zeros(len(confirmed_cases), dtype=float)
    r_h = np.zeros(t, dtype=float)
    d_h = np.zeros(t, dtype=float)
    d_h[0] = float(0.0000001)
    i_h[0] = 0
    # rat
    s_r = np.zeros(t, dtype=float)
    i_r = np.zeros(t, dtype=float)
    res_r = np.zeros(t, dtype=float)
    d_r = np.zeros(t, dtype=float)
    s_r[0] = (s_h * sus_frac) - 20
    res_r[0] = 0.
    i_r[0] = i_r0
    # flea
    searching = 3. / ((s_h * sus_frac) - 20)
    i_f = np.zeros(t, dtype=float)
    fph = np.zeros(t, dtype=float)
    fph[0] = c_cap
    m = 0
    month = 1
    for i, v in enumerate(years_list[1:], 1):
        date_string = v.strftime("%Y-%m-%d")
        temp = data[date_string][0]
        temp_fac = (temp - 15) / 10
        # + rec_r[i - 1]
        N_r = s_r[i - 1] + i_r[i - 1] + res_r[i - 1]
        # natural deaths
        natural_death_unresistant = (s_r[i - 1] * d_rate_ui)
        natural_death_resistant = (res_r[i - 1] * d_rate_ui)
        natural_death_infected = (i_r[i - 1] * d_rate_ui)
        # - Fleas
        if i == 1:
            infected_rat_deaths = d_h[0]
        if fph[i - 1] / c_cap < 1.:
            flea_growth = (g_rate * temp_fac) * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        elif fph[i - 1] / c_cap > 1.:
            flea_growth = -(g_rate * temp_fac) * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        else:
            flea_growth = 0.

        new_infectious = (infected_rat_deaths) * fph[i - 1]
        # could be made temperature dependent
        starvation_deaths = d_rate * i_f[i - 1]
        # number of fleas that find a human
        force_to_humans = min(i_f[i - 1], i_f[i - 1] * np.exp(float(-searching * N_r)))
        # number of fleas that find a rat
        force_to_rats = i_f[i - 1] - force_to_humans
        force_to_rats = force_to_rats * (0.75 - 0.25 * np.tanh(((temp * 9 / 5) + 32) - 80))
        force_to_humans = force_to_humans * 0.9 * (0.75 - 0.25 * np.tanh(((temp * 9 / 5) + 32) - 80))
        fph[i] = fph[i - 1] + flea_growth
        # should add dehydration
        i_f[i] = i_f[i - 1] + new_infectious - starvation_deaths

        # - Rats
        new_infected_rats = beta_r * s_r[i - 1] * force_to_rats / N_r
        new_infected_rats = 0 if new_infected_rats < 0 else new_infected_rats
        new_removed_rats = gamma_r * (i_r[i - 1] - natural_death_infected)
        new_recovered_rats = p_recovery_ur * new_removed_rats
        new_dead_rats = new_removed_rats - new_recovered_rats
        infected_rat_deaths = new_dead_rats

        # born rats
        pressure = N_r / (s_h * sus_frac)
        resistant_born_rats = rep_rate_r * res_r[i - 1] * (inh_res - pressure)
        unresistant_born_rats = ((rep_rate_r * res_r[i - 1] * (1 - inh_res)) + (rep_rate_ur * s_r[i - 1] * (1 - pressure)))

        # time step values
        s_r[i] = min(s_h * sus_frac, s_r[i - 1] + unresistant_born_rats - new_infected_rats - natural_death_unresistant)
        i_r[i] = i_r[i - 1] + new_infected_rats - new_removed_rats - natural_death_infected
        res_r[i] = res_r[i - 1] + new_recovered_rats + resistant_born_rats - natural_death_resistant
        d_r[i] = new_dead_rats + natural_death_unresistant + natural_death_resistant + natural_death_infected

        # - Humans
        N_h = s_h - i_h[i - 1] - r_h[i - 1]
        new_infected_humans = min(s_h, beta_h * N_h * force_to_humans / s_h)
        new_removed_humans = gamma_h * i_h[i - 1]
        new_recovered_humans = p_recovery_h * new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans

        if i == 50:
            i_r[i] = theta
        # time step values
        i_h[i] = i_h[i - 1] + new_infected_humans - new_removed_humans
        r_h[i] = r_h[i - 1] + new_recovered_humans
        d_h[i] = new_dead_humans + 0.0000001
        this_month = int(date_string.split("-")[1])
        if month == this_month:
            i_n[m - 1] += new_infected_humans
        else:
            month = this_month
            m += 1
            if m <= len(i_n):
                i_n[m - 1] += new_infected_humans
            else:
                pass
    return i_n, i_h, r_h, d_h, s_r, i_r, res_r, d_r, i_f, fph

# Likelihood
sim_data = pm.Lambda('sim_data', lambda plague_model=plague_model: plague_model[0])
mortality = pm.Poisson('mortality', mu=sim_data, value=confirmed_cases, observed=True)
mortalitysim = pm.Poisson('mortality_sim', mu=sim_data)