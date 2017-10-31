import numpy as np
import pandas as pd
from datetime import date
import matplotlib.dates as mdates
from tools.load_temp_data import TempReader
import matplotlib.pyplot as plt


def run(beta_h, sus_frac, beta_r, i_r0, inh_res):
    title = "Plague model"
    start_year = 1980
    end_year = 2010
    md = []
    with open("sim_md.csv", mode='r') as file:
        [md.append(float(a)) for a in file.read().split(', ')]
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    # years_list = pd.date_range(date(1990, 1, 1), date(2000, 12, 31)).tolist()
    years_list = pd.date_range(date(start_year, 1, 1), date(end_year, 12, 31)).tolist()

    # -- Params
    data, temp_list = TempReader().cooked()
    t = [x for x in range(0, len(data))]

    # - Human
    beta_h = beta_h
    # .2
    s_h = 25000.
    i_h = np.zeros_like(t)
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
    i_r[0] = i_r0
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

    # temps

    temp_data = np.zeros_like(t)
    temp_data[0] = 23.875

    # -- Simulate
    for i, v in enumerate(years_list[1:], 1):
        temps = data[v.strftime("%Y-%m-%d")]
        temp = temps[0]
        temp_data[i] = temp
        temp_fac = ((temp - 18)/((14/3)*4)) + (1 / 4)
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
        force_to_humans = min(i_f[i - 1], i_f[i - 1] * np.exp(-searching * N_r))
        # number of fleas that find a rat
        force_to_rats = i_f[i - 1] - force_to_humans
        force_to_rats = force_to_rats * (0.75 - 0.25 * np.tanh(((temp * 9/5) + 32) - 80))
        force_to_humans = force_to_humans * (0.75 - 0.25 * np.tanh(((temp * 9/5) + 32) - 80))
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
        unresistant_born_rats = rep_rate_ur * ((res_r[i - 1] * (1 - inh_res)) + (s_r[i - 1] * (1 - pressure)))

        # rec

        # time step values

        s_r[i] = min(s_h * sus_frac, s_r[i - 1] + unresistant_born_rats - new_infected_rats - natural_death_unresistant)
        i_r[i] = i_r[i - 1] + new_infected_rats - new_removed_rats - natural_death_infected
        res_r[i] = res_r[i - 1] + new_recovered_rats + resistant_born_rats - natural_death_resistant
        d_r[i] = new_dead_rats + natural_death_unresistant + natural_death_resistant

        if (i - 125) % 365 == 0 and i_r[i] == 0:
            i_r[i] = 20
            print(i)

        # - Humans
        N_h = s_h + i_h[i - 1] + r_h[i - 1]
        new_infected_humans = min(s_h, beta_h * s_h * force_to_humans / N_h)
        new_removed_humans = gamma_h * i_h[i - 1]
        new_recovered_humans = p_recovery_h * new_removed_humans
        new_dead_humans = new_removed_humans - new_recovered_humans

        # time step values
        i_h[i] = i_h[i - 1] + new_infected_humans - new_removed_humans
        r_h[i] = r_h[i - 1] + new_recovered_humans
        d_h[i] = new_dead_humans + 0.0000001

    md = np.asarray(md, dtype=float)

    # Likelihood
    #mortality = pm.Poisson('mortality', mu=d_h, value=md, observed=True)
    #mortality_sim = pm.Poisson('mortality_sim', mu=d_h)
    # return mortality.logp

    fig, ax = plt.subplots()

    # plot the data
    ax.plot(years_list, d_r, label='dead_rats')
    ax.plot(years_list, s_r, label='susceptible rats')
    ax.plot(years_list, i_r, label="infected rats")
    ax.plot(years_list, res_r, label="resistant rats")
    ax.plot(years_list, d_h, label="I see dead people")
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
    plt.xlabel('time in years')
    plt.ylabel('number of rats')
    plt.show()
    # plt.savefig('SIRD_model.png')

    # with open('sim_md.csv', mode='a') as file:
    #     file.write(", ".join([str(a) for a in d_h.tolist()]) + '\n')


if __name__ == "__main__":
    # make a simple loop for testing priors.
    # beta_h = mf.Distribution('beta_h', lower=1e-9, upper=.2, value=.01)
    # sus_frac = mf.Distribution('fraction', lower=1e-9, upper=1., value=.01)
    # beta_r = mf.Distribution('beta_r', lower=1e-9, upper=1.0, value=.01)
    # i_r0 = mf.Distribution('i_r0', lower=0, upper=18., value=18.)
    # mc = mf.GaussianWalk(10, run, beta_h, sus_frac, beta_r, i_r0)
    # mc.start()
    # 0.975
    run(.1, .12, .08, 0., 0.8)