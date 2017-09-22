import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run():
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    years_list = pd.date_range(date(1990, 1, 1), date(2000, 12, 31)).tolist()

    # -- Params
    temp = [[31.1, 27.1, 23.6, 82], [30.8, 27.2, 23.8, 83], [32.5, 27.5, 23.6, 81], [32.5, 27.4, 22.9, 73],
         [32.0, 26.1, 20.6, 67], [31.0, 24.6, 18.6, 64], [30.8, 24.2, 18.0, 62], [31.4, 24.6, 18.4, 60],
         [32.1, 25.4, 19.6, 63], [32.5, 26.8, 22.0, 66], [32.2, 27.7, 23.5, 72], [31.3, 27.4, 23.7, 80]]
    decade = {}
    for time in years_list:
        time = time.strftime("%Y-%m-%d")
        time_s = time.split("-")
        decade[time] = temp[int(time_s[1])-1]
    t = [x for x in range(0, len(decade))]

    # - Human
    beta_h = .2
    s_h = 25000.
    i_h = np.zeros_like(t)
    r_h = np.zeros_like(t)
    d_h = np.zeros_like(t)
    gamma_h = 0.1
    por_h = .4

    # - rat
    s_r = np.zeros_like(t)
    i_r = np.zeros_like(t)
    rec_r = np.zeros_like(t)
    d_r = np.zeros_like(t)
    sus_frac = .1
    # 0.08
    s_r[0] = s_h * sus_frac
    beta_r = .08
    i_r[0] = 15.
    gamma_r = 0.2
    por_r = .1
    rep_rate_r = .5
    rep_rate_ur = .4
    inh_res = 0.975

    # - flea
    d_rate = 0.2
    # 0.2
    g_rate = .0084
    c_cap = 6.
    i_f = np.zeros_like(t)
    fph = np.zeros_like(t)
    fph[0] = c_cap
    searching = 3. / s_r[0]

    # -- Simulate
    for i in t[1:]:
        N_r = s_r[i - 1] + i_r[i - 1] + rec_r[i - 1]
        # - Fleas
        if i == 1:
            infected_rat_deaths = d_h[0]
            c_cap = fph[0]  # avg number of fleas per rat at carrying capacity
        if fph[i - 1] / c_cap < 1.:
            flea_growth = g_rate * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        elif fph[i - 1] / c_cap > 1.:
            flea_growth = -g_rate * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        else:
            flea_growth = 0.

        new_infectious = infected_rat_deaths * fph[i - 1]
        starvation_deaths = d_rate * i_f[i - 1]
        force_to_humans = min(i_f[i - 1], i_f[i - 1] * np.exp(-searching * N_r))  # number of fleas that find a human
        force_to_rats = i_f[i - 1] - force_to_humans  # number of fleas that find a rat
        fph[i] = fph[i - 1] + flea_growth
        i_f[i] = i_f[i - 1] + new_infectious - starvation_deaths

        # - Rats
        new_infected_rats = min(s_r[i - 1], beta_r * s_r[i - 1] * force_to_rats / N_r)
        new_removed_rats = gamma_r * i_r[i - 1]
        new_recovered_rats = por_r * new_removed_rats
        new_dead_rats = new_removed_rats - new_recovered_rats
        infected_rat_deaths = new_dead_rats

        # born rats
        resistant_born_rats = (rep_rate_r * rec_r[i - 1] * (inh_res - (N_r / s_h)))
        resistant_born_rats = 0 if N_r / s_h < 0 else resistant_born_rats
        unresistant_from_resistant = (rep_rate_ur * rec_r[i - 1] * (1. - inh_res))
        unresistant_born_rats = (rep_rate_ur * s_r[i - 1] * (1. - (N_r / s_h)))
        unresistant_born_rats = 0 if N_r / s_h < 0 else unresistant_born_rats
        born_rats = unresistant_born_rats + unresistant_from_resistant

        # natural deaths
        natural_death_unresistant = (s_r[i - 1] * d_rate)
        natural_death_resistant = (rec_r[i - 1] * d_rate)

        # time step values
        s_r[i] = s_r[i - 1] + born_rats - new_infected_rats - natural_death_unresistant
        i_r[i] = i_r[i - 1] + new_infected_rats - new_removed_rats
        rec_r[i] = rec_r[i - 1] + new_recovered_rats + resistant_born_rats - natural_death_resistant
        d_r[i] = new_dead_rats + natural_death_unresistant + natural_death_resistant

        # - Humans
        N_h = s_h + i_h[i - 1] + r_h[i - 1]
        new_infected_humans = min(s_h, beta_h * s_h * force_to_humans / N_h)
        new_removed_humans = gamma_h * i_h[i - 1]
        new_recovered_humans = por_h * new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans

        # time step values
        i_h[i] = i_h[i - 1] + new_infected_humans - new_removed_humans
        r_h[i] = r_h[i - 1] + new_recovered_humans
        d_h[i] = new_dead_humans

    fig, ax = plt.subplots()

    # plot the data
    ax.plot(years_list, d_r, label='dead_rats')
    ax.plot(years_list, s_r, label='susceptible rats')
    ax.plot(years_list, i_r, label="infected rats")
    ax.plot(years_list, rec_r, label="resistant rats")

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
    plt.xlabel('time in years')
    plt.ylabel('number of rats')
    # plt.savefig('SIRD_model.png')
    plt.show()

