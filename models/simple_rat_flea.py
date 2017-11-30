import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
from tools.TempReader import TempReader
import matplotlib.pyplot as plt
import multiprocessing as mp


def shrew_presence(x, pop, frac):
    return (pop + (pop * frac) * np.cos((x / (365. / (2. * np.pi))) - (132. / (365. / (2. * np.pi)))))/pop


def shrew_flea_index(x):
    return 5. + 4.5 * np.cos((x / (365. / (2. * np.pi))) - (227. / (365. / (2. * np.pi))))


def p_t(x, pop):
    return shrew_presence(x, pop, 0.2) * (shrew_flea_index(x)/9.5)


def day_of_year(x):
    return x.timetuple().tm_yday


def run(N_h, beta_h, rat_pop, beta_r, i_rpd, inh_res, years_list, months_list, shrew_pop, temp_scale):
    confirmed_cases = [0, 8, 12, 62, 16, 2, 14, 6, 5, 0, 0, 0, 0, 1, 5, 22, 39, 11, 8, 5, 6, 2, 1, 0, 0, 10, 38, 59, 74,
                       13, 6, 1, 1, 0, 0, 0, 0, 4, 17, 18, 29, 9, 8, 3, 3, 1, 0, 1, 0]
    scaled_cases = [0.0, 52.0, 78.0, 403.0, 104.0, 13.0, 91.0, 36.0, 30.0, 0.0, 0.0, 0.0, 0.0, 6.0, 30.0, 132.0, 234.0,
                    66.0, 48.0, 15.0, 18.0, 6.0, 3.0, 0.0, 0.0, 30.0, 114.0, 177.0, 222.0, 39.0, 18.0, 3.0, 3.0, 0.0,
                    0.0, 0.0, 0.0, 12.0, 51.0, 54.0, 87.0, 27.0, 24.0, 24.0, 24.0, 8.0, 0.0, 8.0, 0.0]
    # -- Params
    data, temp_list = TempReader().cooked()
    t = [x for x in range(0, len(years_list))]
    # - Human
    i_h = np.zeros_like(t)
    r_h = np.zeros_like(t)
    d_h = np.zeros_like(t, dtype=float)
    d_h[0] = 2.0
    i_h[0] = 1.0
    gamma_h = 0.1
    p_recovery_h = .4
    # - rat
    s_r = np.zeros_like(t)
    i_r = np.zeros_like(t)
    i_r[0] = 0
    # rec_r = np.zeros_like(t)
    res_r = np.zeros_like(t)
    d_r = np.zeros_like(t)
    s_r[0] = rat_pop
    beta_r = beta_r
    gamma_r = 0.2
    p_recovery_ur = .1
    rep_rate_r = .4 * (1 - 0.234)
    rep_rate_ur = .4
    inh_res = inh_res
    d_rate_ui = 1./365.
    # - flea
    d_rate = 0.2
    g_rate = .0084
    c_cap = 6.
    i_f = np.zeros_like(t)
    fph = np.zeros_like(t)
    fph[0] = c_cap
    searching = 3. / (s_r[0] + res_r[0])
    # shrews
    # -- Simulate
    for i, v in enumerate(years_list[1:], 1):
        # shrews
        if p_t(day_of_year(v), shrew_pop) >= 1:
            shrew_transference = i_rpd
        else:
            shrew_transference = 0
        # temperature data and factor calculations
        date_string = v.strftime("%Y-%m-%d")
        temp = data[date_string][0] * temp_scale
        temp_growth_factor = (temp - 15.0) / 10.0
        temp_spread_factor = (0.75 - 0.25 * np.tanh(((temp * 9. / 5.) + 32.) - 80.))
        # the total amount of rats
        N_r = s_r[i - 1] + i_r[i - 1] + res_r[i - 1]
        # - Fleas
        if i == 1:
            infected_rat_deaths = d_h[0]
        if fph[i - 1] / c_cap < 1.:
            flea_growth = (g_rate * temp_growth_factor) * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        elif fph[i - 1] / c_cap > 1.:
            flea_growth = -(g_rate * temp_growth_factor) * (fph[i - 1] * (1. - (fph[i - 1] / c_cap)))
        else:
            flea_growth = 0.
        new_infectious = infected_rat_deaths * fph[i - 1]
        starvation_deaths = d_rate * i_f[i - 1]
        # number of fleas that find a human
        force_to_humans = min(i_f[i - 1], i_f[i - 1] * np.exp(float(-searching * N_r)))
        # number of fleas that find a rat
        force_to_rats = i_f[i - 1] - force_to_humans
        force_to_rats = force_to_rats * temp_spread_factor
        force_to_humans = force_to_humans * temp_spread_factor
        fph[i] = fph[i - 1] + flea_growth
        # should add dehydration
        i_f[i] = i_f[i - 1] + new_infectious - starvation_deaths

        # - Rats
        # natural deaths
        natural_death_unresistant = s_r[i - 1] * d_rate_ui
        natural_death_resistant = res_r[i - 1] * d_rate_ui
        natural_death_infected = i_r[i - 1] * d_rate_ui
        # --
        new_infected_rats = min(s_r[i - 1], beta_r * s_r[i - 1] * (force_to_rats / N_r))
        new_removed_rats = gamma_r * (i_r[i - 1] - natural_death_infected)
        new_recovered_rats = p_recovery_ur * new_removed_rats
        new_dead_rats = new_removed_rats - new_recovered_rats
        infected_rat_deaths = new_dead_rats
        # born rats
        pressure = N_r / rat_pop
        resistant_born_rats = max(0, rep_rate_r * res_r[i - 1] * (inh_res - pressure))
        unresistant_born_rats = max(0, (rep_rate_r * res_r[i - 1] * (1 - inh_res)) + (rep_rate_ur * s_r[i - 1] * (1 - pressure)))
        # time step values
        s_r[i] = s_r[i - 1] + unresistant_born_rats - new_infected_rats - natural_death_unresistant - shrew_transference
        i_r[i] = i_r[i - 1] + new_infected_rats - new_removed_rats - natural_death_infected + shrew_transference
        res_r[i] = res_r[i - 1] + new_recovered_rats + resistant_born_rats - natural_death_resistant
        d_r[i] = new_dead_rats + natural_death_unresistant + natural_death_resistant + natural_death_infected
        # - Humans
        s_h = N_h - i_h[i - 1] - r_h[i - 1]
        new_infected_humans = min(N_h, beta_h * s_h * force_to_humans / N_h)
        new_removed_humans = gamma_h * i_h[i - 1]
        new_recovered_humans = p_recovery_h * new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans
        # time step values
        i_h[i] = i_h[i - 1] + new_infected_humans - new_removed_humans
        r_h[i] = r_h[i - 1] + new_recovered_humans
        d_h[i] = new_dead_humans + 0.0000001
    graphs = [[months_list, years_list, {"infected humans": i_h.tolist(),
                                       "Conf. lab cases": confirmed_cases,
                                       "Scaled cases": scaled_cases}],
            [months_list, years_list, {"susceptible rats": s_r.tolist(),
                                       "infected rats": i_r.tolist(),
                                       "resistant rats": res_r.tolist()}]]
    jobs = []
    for g in graphs:
        p = mp.Process(target=plot, args=g)
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()


def plot(months_list, years_list, kwargs):
    title = "Plague model"
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    fig, ax = plt.subplots()

    # plot the data
    for label, data in kwargs.items():
        if len(data) == len(years_list):
            ax.plot(years_list, data, label=label)
        else:
            ax.plot(months_list, data, label=label)


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
    start = [1995, 6, 1]
    end = [1999, 7, 1]
    months_list = pd.date_range(datetime.date(start[0], start[1], start[2]), datetime.date(end[0], end[1], end[2]), freq='M').tolist()
    years_list = pd.date_range(datetime.date(start[0], start[1], start[2]), datetime.date(end[0], end[1], end[2])).tolist()
    run(20000., .15, 8000.0, .15, 2.0, 0.975, years_list, months_list, 12000., 1.085)
    # run(N_h, beta_h, rat_pop, beta_r, i_rpd, inh_res)
