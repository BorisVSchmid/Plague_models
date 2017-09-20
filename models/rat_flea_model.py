import numpy as np
import scipy as sp
import pymc3 as pm

def run():
    with pm.Model() as plague_model:
        # -- Params
        t = range(1, 3651, 1)
        # - Human
        beta_h = pm.Uniform(1e-9, .2, value=.2)
        s_h = 100000.
        i_h = np.zeros(t, dtype=object)
        r_h = np.zeros(t, dtype=object)
        d_h = np.zeros(t, dtype=object)
        gamma_h = .1
        por_h = .4
        # - rat
        s_r = np.zeros(t, dtype=object)
        i_r = np.zeros(t, dtype=object)
        res_r = np.zeros(t, dtype=object)
        rec_r = np.zeros(t, dtype=object)
        d_r = np.zeros(t, dtype=object)
        sus_frac = pm.Uniform(1e-9, 1., value=.08)
        s_r[0] = s_h * sus_frac
        beta_r = pm.Uniform(1e-9, 1.0, value=.08)
        i_r[0] = pm.Uniform(0.15, 1.)
        gamma_r = 1/5.2
        por_r = .1
        # - flea
        d_rate = .2
        g_rate = .0084
        c_cap = 6.
        i_f = np.zeros(t, dtype=object)
        fph = np.zeros(t, dtype=object)
        fph[0] = c_cap
        searching = 3. / s_r[0]
        # -- Simulate
        for i in t:
            N_r = s_r[i - 1] + i_r[i - 1] + rec_r[i - 1]
            if i == 1:
                infected_rat_deaths = d_h[0]
                # Fleas
                c_cap = fph[0]  # avg number of fleas per rat at carrying capacity
            if fph[i - 1] / c_cap < 1.:
                flea_growth = g_rate * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
            elif fph[i - 1] / c_cap > 1.:
                flea_growth = -g_rate * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
            else:
                flea_growth = 0.
            new_infectious = infected_rat_deaths * (fph[i - 1])
            starvation_deaths = d_rate * i_f[i - 1]
            force_to_humans = min(i_f[i - 1], i_f[i - 1] * np.exp(-searching * N_r))  # number of fleas that find a human
            force_to_rats = i_f[i - 1] - force_to_humans  # number of fleas that find a rat
            fph[i] = fph[i - 1] + flea_growth
            i_f[i] = i_f[i - 1] + new_infectious - starvation_deaths
            # Rats
            new_infected_rats = min(s_r[i - 1], beta_r * s_r[i - 1] * force_to_rats / N_r)
            new_removed_rats = gamma_r * i_r[i - 1]
            new_recovered_rats = por_r * new_removed_rats
            new_dead_rats = new_removed_rats - new_recovered_rats
            infected_rat_deaths = new_dead_rats
            s_r[i] = s_r[i - 1] - new_infected_rats
            i_r[i] = i_r[i - 1] + new_infected_rats - new_removed_rats
            rec_r[i] = rec_r[i - 1] + new_recovered_rats
            d_r[i] = new_dead_rats
            # Humans
            N_h = s_h + i_h[i - 1] + r_h[i - 1]
            new_infected_humans = min(s_h, beta_h * s_h * force_to_humans / N_h)
            new_removed_humans = gamma_h * i_h[i - 1]
            new_recovered_humans = por_h * new_removed_humans
            new_dead_humans = new_removed_humans - new_recovered_humans
            i_h[i] = i_h[i - 1] + new_infected_humans - new_removed_humans
            r_h[i] = r_h[i - 1] + new_recovered_humans
            d_h[i] = new_dead_humans

    with plague_model:
        nuts_trace = pm.sample(1000)

    pm.traceplot(nuts_trace)
    pm.summary(nuts_trace)



