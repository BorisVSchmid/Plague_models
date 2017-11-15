import numpy as np
import pandas as pd
from datetime import date
import matplotlib.dates as mdates
from tools.TempReader import TempReader
from os.path import dirname, abspath
import os.path
import matplotlib.pyplot as plt
import time


def run(s_h, beta_h, sus_frac, beta_r, i_r0, inh_res):
    title = "Plague model"
    dir = dirname(dirname(abspath(__file__)))
    start_year = 1995
    end_year = 1999
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    years_list = pd.date_range(date(start_year, 1, 1), date(end_year, 7, 1)).tolist()
    months_list = pd.date_range(date(start_year, 1, 1), date(end_year, 7, 1), freq='M').tolist()
    confirmed_cases = [0, 0, 1, 0, 2, 0, 8, 12, 62, 16, 2, 14, 6, 5, 0, 0, 0, 0, 1, 5, 22, 39, 11, 8, 5, 6, 2, 1, 0, 0,
                       10, 38, 59, 74, 13, 6, 1, 1, 0, 0, 0, 0, 4, 17, 18, 29, 9, 8, 3, 3, 1, 0, 1, 0]
    # -- Params
    data, temp_list = TempReader().cooked()
    t = [x for x in range(0, len(data))]

    # - Human
    beta_h = beta_h
    # .2
    s_h = s_h
    i_h = np.zeros_like(t)
    i_n = np.zeros_like(confirmed_cases, dtype=float)
    r_h = np.zeros_like(t)
    d_h = np.zeros_like(t, dtype=float)
    d_h[0] = 0.0000001
    i_h[0] = 2
    gamma_h = 0.1
    p_recovery_h = .4

    # - rat
    s_r = np.zeros_like(t)
    i_r = np.zeros_like(t)
    # rec_r = np.zeros_like(t)
    res_r = np.zeros_like(t)
    d_r = np.zeros_like(t)
    sus_frac = sus_frac
    # 0.08
    s_r[0] = (s_h * sus_frac) - 20
    beta_r = beta_r
    # .08
    gamma_r = 0.2
    # .2
    # .1 \/
    p_recovery_ur = .1
    rep_rate_r = .4 * (1 - 0.234)
    rep_rate_ur = .4
    inh_res = inh_res
    d_rate_ui = 1/(365 * 1)

    # - flea
    d_rate = 0.2
    # 0.2
    g_rate = .0084
    c_cap = 6.
    i_f = np.zeros_like(t)
    fph = np.zeros_like(t)
    fph[0] = c_cap
    searching = 3. / (s_r[0] + res_r[0])
    m = 0
    month = 1
    # -- Simulate
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
            i_r[i] = i_r0

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

    # with open(os.path.join(dir, 'data', 'sim_md.csv'), mode='w') as file:
    #     file.write(", ".join([str(a) for a in d_h.tolist()]) + '\n')
    fig, ax = plt.subplots()

    # plot the data
    ax.plot(months_list, i_n, label='infected humans')
    ax.plot(months_list, confirmed_cases, label='confirmed cases')
    # ax.plot(years_list, temp_data, label="temperature data")

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    # set the axis limit
    datemin = min(years_list)
    datemax = max(years_list) + 1
    ax.set_xlim(datemin, datemax)

    # format the coords message box
    def price(x):
        return '$%1.2f' % x
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = price
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    # some extra plot formating
    ax.legend(loc='best')
    plt.style.use('ggplot')
    plt.rc('font', size=16)
    plt.rc('lines', linewidth=2)
    plt.rc('figure', autolayout=True)
    plt.title(title)
    plt.xlabel('time in months')
    plt.ylabel('number of humans')
    plt.show()
    plt.close()


if __name__ == "__main__":
    for x in range(15, 25, 2):
        run(20000, .01, .10, .085, x, 0.8)
    # run(s_h, beta_h, sus_frac, beta_r, i_r0, inh_res)