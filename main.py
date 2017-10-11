from models.pymc2_test import *
import pymc as pm
from pymc.Matplot import plot
import matplotlib.pyplot as plt
import scipy.stats


# temp = [[31.1, 27.1, 23.6, 82], [30.8, 27.2, 23.8, 83], [32.5, 27.5, 23.6, 81], [32.5, 27.4, 22.9, 73],
#         [32.0, 26.1, 20.6, 67], [31.0, 24.6, 18.6, 64], [30.8, 24.2, 18.0, 62], [31.4, 24.6, 18.4, 60],
#         [32.1, 25.4, 19.6, 63], [32.5, 26.8, 22.0, 66], [32.2, 27.7, 23.5, 72], [31.3, 27.4, 23.7, 80]]


if __name__ == "__main__":
    vars = [md, beta, s_h, gamma_h, p_recovery_h, fraction, sigma, i_r0, s_r0, gamma_r,
            p_recovery_ur, rep_rate_r, rep_rate_ur, inh_res, d_rate_ui, d_rate, g_rate, c_cap,
            fph0, searching, d_, mortality, mortality_sim]

    mc = pm.MCMC(vars, db='hdf5', dbname='rat.hdf5', dbcomplevel=9, dbcomplib='bzip2')
    mc.use_step_method(pm.AdaptiveMetropolis, [beta, fraction])
    mc.sample(iter=1000, burn=500, thin=2, verbose=1)
    mc.summary()
    M = pm.MAP(mc)
    mc.db.close()
    print('fit')
    M.fit(method='fmin_powell')
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

