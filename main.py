from models.pymc2_test import *
import pymc as pm
from pymc.Matplot import plot
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import numpy as np


def set_time_period(start, finish):
    years_list = pd.date_range(start, finish).tolist()

    # -- Params
    temp = np.array([[31.1, 27.1, 23.6], [30.8, 27.2, 23.8], [32.5, 27.5, 23.6], [32.5, 27.4, 22.9], [32.0, 26.1, 20.6],
                    [31.0, 24.6, 18.6], [30.8, 24.2, 18.0], [31.4, 24.6, 18.4], [32.1, 25.4, 19.6], [32.5, 26.8, 22.0],
                    [32.2, 27.7, 23.5], [31.3, 27.4, 23.7]])

    with open('time.date', 'w') as file:
        for time in years_list:
            time = time.strftime("%Y-%m-%d")
            file.write('{} : {}\n'.format(time, ', '.join([str(x) for x in temp[int(time.split("-")[1]) - 1]])))
        return years_list

if __name__ == "__main__":
    # years_list = set_time_period(date(1990, 1, 1), date(1990, 12, 31))
    vars = [md, beta, s_h, gamma_h, p_recovery_h, fraction, sigma, i_r0, s_r0, gamma_r,
            p_recovery_ur, rep_rate_r, rep_rate_ur, inh_res, d_rate_ui, d_rate, g_rate, c_cap,
            fph0, searching, d_, mortality, mortality_sim]

    mc = pm.MCMC(vars)
    mc.use_step_method(pm.AdaptiveMetropolis, [beta, fraction])
    mc.sample(iter=1000, burn=500, thin=2, verbose=1)
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
    plt.plot(mortality_sim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
    y_min = mortality_sim.stats()['quantiles'][2.5]
    y_max = mortality_sim.stats()['quantiles'][97.5]
    plt.fill_between(range(0, len(md)), y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
    plt.legend()
    plt.show()

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

