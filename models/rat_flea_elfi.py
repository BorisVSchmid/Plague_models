"""This model is the rat flea human with seasonality model made to be fitted against mortality data"""
import elfi
import numpy as np
import pandas as pd
from datetime import date
from tools.TempReader import TempReader
from os.path import dirname, abspath
import os.path


# -- Simulate
def plague_model(s_h, beta_h, sus_frac, beta_r, inh_res, batch_size=1, random_state=None):
    dir = dirname(dirname(abspath(__file__)))
    # time span
    start_year = 1991
    end_year = 1991
    md = np.array(list(map(float, open(os.path.join(dir, 'data', 'sim_md1.csv'), 'r').read().split(', '))), dtype=float)
    years_list = pd.date_range(date(start_year, 1, 1), date(end_year, 12, 31)).tolist()
    # -- Params
    data, temp_list = TempReader().cooked()
    t = len(data)
    # - human
    # .2
    gamma_h = 0.1
    p_recovery_h = .4
    # - rat
    res_r0 = 0.
    # .08
    # 0.08
    # .2
    gamma_r = 0.2
    # .1
    p_recovery_ur = .1
    rep_rate_r = .4 * (1 - 0.234)
    rep_rate_ur = .4
    d_rate_ui = 1 / (365 * 1)
    # - flea
    d_rate = 0.2
    # 0.2
    g_rate = .0084
    c_cap = 6.
    searching = 3. / ((s_h[:] * sus_frac[:]) - 20 + res_r0)
    #human
    i_h = np.asanyarray(np.zeros(t, dtype=float))
    r_h = np.zeros(t, dtype=float)
    d_h = np.zeros(t, dtype=float)
    d_h[0] = 0.0000001
    i_h[0] = 2
    # rat
    s_r = np.zeros(t, dtype=float)
    i_r = np.zeros(t, dtype=float)
    res_r = np.zeros(t, dtype=float)
    d_r = np.zeros(t, dtype=float)
    s_r[0] = (s_h * sus_frac) - 20
    res_r[0] = res_r0
    # flea
    i_f = np.zeros(t, dtype=float)
    fph = np.zeros(t, dtype=float)
    fph[0] = c_cap
    for i, v in enumerate(years_list[1:], 1):
        temps = data[v.strftime("%Y-%m-%d")]
        temp = temps[0]
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

        # if (i - 125) % 365 == 0 and i_r[i] == 0:
        if i == 125:
            i_r[i] = 20

        # - Humans
        N_h = s_h - i_h[i - 1] - r_h[i - 1]
        new_infected_humans = min(s_h, beta_h * N_h * force_to_humans / s_h)
        new_removed_humans = gamma_h * i_h[i - 1]
        new_recovered_humans = p_recovery_h * new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans

        # time step values
        i_h[i] = i_h[i - 1] + new_infected_humans - new_removed_humans
        r_h[i] = r_h[i - 1] + new_recovered_humans
        d_h[i] = new_dead_humans + 0.0000001
    return d_h


if __name__ == '__main__':
    s_h = elfi.Prior('uniform', 15000.0, 30000.0)
    beta_h = elfi.Prior('uniform', 0.1, .2)
    sus_frac = elfi.Prior('uniform', 0.01, 0.99)
    beta_r = elfi.Prior('uniform', 1e-3, .199)
    inh_res = elfi.Prior('uniform', 0.5, 0.475)


