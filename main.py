from models.rat_flea_pymc import *
import pymc as pm
from pymc.Matplot import plot
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import matplotlib.dates as mdates


if __name__ == "__main__":
    title = "Plague model"
    years_list = pd.date_range(date(1991, 1, 1), date(1991, 12, 31)).tolist()
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    vars = [md, sigma, beta, gamma_h, p_recovery_h, phi, rho, gamma_r, p_recovery_ur,
           rep_rate_r, rep_rate_ur, iota, d_rate_ui, d_rate, g_rate, c_cap, searching, d,
           mortality, mortalitysim]
    mc = pm.MCMC(vars, db='hdf5', dbname='ratflea.hdf5')
    mc.use_step_method(pm.AdaptiveMetropolis, [sigma, beta, phi, rho, iota])
    mc.sample(iter=4, verbose=4)
    mc.summary()
    M = pm.MAP(mc)
    print('fit')
    M.fit(method='fmin')
    M.BIC
    plot(mc)
    plt.figure(figsize=(10, 10))
    plt.title("Plague Mahajanga")
    plt.xlabel('Day')
    plt.ylabel('Deaths')
    plt.plot(md, 'o', mec='black', color='black', label='Simulated data')
    plt.plot(mortalitysim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
    y_min = mortalitysim.stats()['quantiles'][2.5]
    y_max = mortalitysim.stats()['quantiles'][97.5]
    plt.fill_between(range(0, len(md)), y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
    plt.legend()
    plt.show()

    # data, temp_list = TempLoader().read_raw()
    # data, temp_list = TempLoader(start=1980, end=2010, update=True, floc="data",
    #                              fname="1112204").read_raw()

    # fig, ax = plt.subplots()
    #
    # # plot the data
    # ax.plot(years_list, temp_list, label="temperature data")
    #
    # # format the ticks
    # ax.xaxis.set_major_locator(years)
    # ax.xaxis.set_major_formatter(yearsFmt)
    # ax.xaxis.set_minor_locator(months)
    #
    # # set the axis limit
    # datemin = min(years_list)
    # datemax = max(years_list) + 1
    # ax.set_xlim(datemin, datemax)
    #
    #
    # # format the coords message box
    # def price(x):
    #     return '$%1.2f' % x
    #
    #
    # ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    # ax.format_ydata = price
    # ax.grid(True)
    #
    # # rotates and right aligns the x labels, and moves the bottom of the
    # # axes up to make room for them
    # fig.autofmt_xdate()
    #
    # # some extra plot formating
    # ax.legend(loc='best')
    # plt.style.use('ggplot')
    # plt.rc('font', size=16)
    # plt.rc('lines', linewidth=2)
    # plt.rc('figure', autolayout=True)
    # plt.title(title)
    # plt.xlabel('time in years')
    # plt.ylabel('number of rats')
    # plt.show()

    # vars = [md, beta, s_h, gamma_h, p_recovery_h, fraction, sigma, i_r0, s_r0, gamma_r,
    #         p_recovery_ur, rep_rate_r, rep_rate_ur, inh_res, d_rate_ui, d_rate, g_rate, c_cap,
    #         fph0, searching, d_, mortality, mortality_sim]
    #
    # mc = pm.MCMC(vars)
    # mc.use_step_method(pm.AdaptiveMetropolis, [beta, fraction])
    # mc.sample(iter=200, verbose=1)
    # mc.summary()
    # M = pm.MAP(mc)
    # print('fit')
    # M.fit(method='fmin')
    # M.BIC
    # plot(mc)
    # plt.figure(figsize=(10, 10))
    # plt.title("Plague Mahajanga")
    # plt.xlabel('Day')
    # plt.ylabel('Deaths')
    # plt.plot(md, 'o', mec='black', color='black', label='Simulated data')
    # plt.plot(mortality_sim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
    # y_min = mortality_sim.stats()['quantiles'][2.5]
    # y_max = mortality_sim.stats()['quantiles'][97.5]
    # plt.fill_between(range(0, len(md)), y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
    # plt.legend()
    # plt.show()

# from models.an_model import *
# import pymc as pm
# import matplotlib.pyplot as plt
# from pymc.Matplot import plot
#
#
# if __name__ == "__main__":
#     vars = [beta_r, gamma, sigma, searching, H_f0, I_r0, R_r0, D_r0, S_h0,
#             I_h0, D_h0, D_h, beta_h, pop_size, sus_frac, mortality,
#             mortality_data, mortality_sim]
#     mc = pm.MCMC(vars)
#     mc.use_step_method(pm.AdaptiveMetropolis, [beta_r, beta_h, sus_frac, I_r0])
#     mc.sample(iter=1000, burn=500, thin=2, verbose=1)
#     mc.summary()
#     mc.DIC
#
#     plot(mc)
#     plt.figure(figsize=(10, 10))
#     plt.title('Barcelona 1490')
#     plt.xlabel('Day')
#     plt.ylabel('Deaths')
#     plt.plot(mortality_data, 'o', mec='black', color='black', label='Observed deaths')
#     plt.plot(mortality_sim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
#     y_min = mortality_sim.stats()['quantiles'][2.5]
#     y_max = mortality_sim.stats()['quantiles'][97.5]
#     plt.fill_between(range(0, len(mortality_data)), y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
#     plt.legend()
#     plt.show()

