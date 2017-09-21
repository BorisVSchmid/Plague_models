import numpy as np
import matplotlib.pyplot as plt


def run():
    # -- Params
    t = [i for i in range(1001)]
    # - Human
    beta_h = .2
    s_h = 100000.
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
    sus_frac = .08
    s_r[0] = s_h * sus_frac
    beta_r = .08
    i_r[0] = 15.
    gamma_r = 0.2
    por_r = .1
    rep_rate = .2
    inh_res = 0.975
    # - flea
    d_rate = 0.2
    g_rate = .0084
    c_cap = 6.
    i_f = np.zeros_like(t)
    fph = np.zeros_like(t)
    fph[0] = c_cap
    searching = 3. / s_r[0]
    week = 0
    # -- Simulate
    for i in t[1:]:
        N_r = s_r[i - 1] + i_r[i - 1] + rec_r[i - 1]
        # Fleas
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
        # Rats
        new_infected_rats = min(s_r[i - 1], beta_r * s_r[i - 1] * force_to_rats / N_r)
        new_removed_rats = gamma_r * i_r[i - 1]
        new_recovered_rats = por_r * new_removed_rats
        new_dead_rats = new_removed_rats - new_recovered_rats
        infected_rat_deaths = new_dead_rats
        born_rats = (rep_rate * s_r[i - 1] * (1. - (N_r/c_cap))) + (rep_rate * rec_r[i - 1] * (1. - inh_res))
        s_r[i] = (rep_rate * s_r[i - 1]) - new_infected_rats - (s_r[i - 1] * d_rate) + born_rats
        i_r[i] = i_r[i - 1] + new_infected_rats - new_removed_rats
        rec_r[i] = rec_r[i - 1] + new_recovered_rats + (rep_rate * rec_r[i - 1] * (inh_res - (N_r/c_cap))) - (d_rate * rec_r[i - 1])
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
        if i % 7 == 0 and i != 0:
            week += 1
            print("{} dead humans in week{} and {} dead rats".format(str(sum(d_h[i-7:i])), week, str(sum(d_r[i-7:i]))))
    plt.style.use('ggplot')
    plt.rc('font', size=16)
    plt.rc('lines', linewidth=2)
    plt.rc('figure', autolayout=True)
    plt.xlabel('time')
    plt.ylabel('number of people')
    plt.plot(d_r, label='dead_rats')
    plt.plot(s_r, label='rats')
    plt.plot(i_r, label="infected rats")
    plt.legend(loc='best')
    # plt.savefig('SIRD_model.png')
    plt.show()
