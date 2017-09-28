import numpy as np
import scipy as sp
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import pymc3 as pm
import pandas as pd
from datetime import date
# import matplotlib.dates as mdates

# theano config
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.compute_test_value = 'warn'

new_

def run():
    with pm.Model() as plague_model:
        # years = mdates.YearLocator()  # every year
        # months = mdates.MonthLocator()  # every month
        # yearsFmt = mdates.DateFormatter('%Y')
        years_list = pd.date_range(date(1990, 1, 1), date(2000, 12, 31)).tolist()

        # -- Params
        temp = [[31.1, 27.1, 23.6, 82], [30.8, 27.2, 23.8, 83], [32.5, 27.5, 23.6, 81], [32.5, 27.4, 22.9, 73],
                [32.0, 26.1, 20.6, 67], [31.0, 24.6, 18.6, 64], [30.8, 24.2, 18.0, 62], [31.4, 24.6, 18.4, 60],
                [32.1, 25.4, 19.6, 63], [32.5, 26.8, 22.0, 66], [32.2, 27.7, 23.5, 72], [31.3, 27.4, 23.7, 80]]
        decade = {}
        for time in years_list:
            time = time.strftime("%Y-%m-%d")
            time_s = time.split("-")
            decade[time] = temp[int(time_s[1]) - 1]
        t = [x for x in range(0, len(decade))]
        # - Human
        beta_h = pm.Uniform(lower=1e-9, upper=.2, name="beta_h")
        i_h = np.zeros_like(t, dtype=object)
        r_h = np.zeros_like(t, dtype=object)
        d_h = np.zeros_like(t, dtype=object)
        s_h = 25000.
        gamma_h = 0.1
        p_recovery_h = .4

        # - rat
        sus_frac = pm.Uniform(lower=1e-9, upper=1., name="sus_frac")
        beta_r = pm.Uniform(lower=1e-9, upper=1.0, name="beta_r")
        # 0.08
        # rec_r[0] = s_h * sus_frac
        gamma_r = 0.2
        # .1 \/
        p_recovery_ur = .058
        rep_rate_r = .4 * (1 - 0.234)
        rep_rate_ur = .4
        inh_res = 0.975
        d_rate_ui = 1 / (365 * 1)

        # - flea

        i_f = np.zeros_like(t, dtype=object)
        fph = np.zeros_like(t, dtype=object)
        d_rate = 0.2
        # 0.2
        g_rate = .0084
        c_cap = 6.
        fph[0] = c_cap
        searching = 3. / (s_r + res_r[0])

        # -- functions and conditions
        # a, n = T.dscalars('a', 'n')
        # a.tag.test_value = 1.
        # n.tag.test_value = 0.
        # non_negative = ifelse(T.lt(a, n), n, a)
        # f_non_negative = theano.function([a, n], non_negative,
        #                                  mode=theano.Mode(linker='vm'))
        # -- Simulate
        # + rec_r
        N_r = s_r + i_r + res_r
        # - Fleas
        flea_growth = g_rate * (fph * (1. - (fph / c_cap)))

        new_infectious = infected_rat_deaths * fph
        starvation_deaths = d_rate * i_f
        # number of fleas that find a human
        force_to_humans = i_f * np.exp(-searching * N_r)
        # force_to_humans = f_non_negative([force_to_humans], 0.)
        # number of fleas that find a rat
        force_to_rats = i_f - force_to_humans
        fph = fph + flea_growth
        i_f = i_f + new_infectious - starvation_deaths

        # - Rats
        new_infected_rats = beta_r * s_r * force_to_rats / N_r
        new_removed_rats = gamma_r * i_r
        new_recovered_rats = p_recovery_ur * new_removed_rats
        new_dead_rats = new_removed_rats - new_recovered_rats
        infected_rat_deaths = new_dead_rats

        # born rats
        resistant_born_rats = (rep_rate_r * res_r * (inh_res - (N_r / (s_h * sus_frac))))
        unresistant_from_resistant = (rep_rate_ur * res_r * (1. - inh_res))
        unresistant_born_rats = (rep_rate_ur * (res_r + s_r) * (1. - (N_r / (s_h * sus_frac))))
        born_rats = unresistant_born_rats + unresistant_from_resistant

        # natural deaths
        natural_death_unresistant = (s_r * d_rate_ui)
        natural_death_resistant = (res_r * d_rate_ui)
        # rec

        # time step values
        s_r = s_r + born_rats - new_infected_rats - natural_death_unresistant
        i_r = i_r + new_infected_rats - new_removed_rats
        # rec_r = rec_r + new_recovered_rats - natural_death_recovered
        res_r = res_r + new_recovered_rats + resistant_born_rats - natural_death_resistant
        d_r = new_dead_rats + natural_death_unresistant + natural_death_resistant

        # - Humans
        N_h = s_h + i_h + r_h
        new_infected_humans = min(s_h, beta_h * s_h * force_to_humans / N_h)
        new_removed_humans = gamma_h * i_h
        new_recovered_humans = p_recovery_h * new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans

        # time step values
        i_h = i_h + new_infected_humans - new_removed_humans
        r_h = r_h + new_recovered_humans
        d_h = new_dead_humans

    with plague_model:
        nuts_trace = pm.sample(1000)

    pm.traceplot(nuts_trace)
    pm.summary(nuts_trace)
