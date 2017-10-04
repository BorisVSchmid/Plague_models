from models.pymc2_test import *
import pymc as pm
import matplotlib.pyplot as plt
from pymc.Matplot import plot
import scipy.stats

if __name__ == "__main__":
    vars = [md, beta, s_h, gamma_h, p_recovery_h, fraction, sigma, i_r0, s_r0, gamma_r,
            p_recovery_ur, rep_rate_r, rep_rate_ur, inh_res, d_rate_ui, d_rate, g_rate, c_cap,
            fph0, searching, d_, mortality, mortality_sim]

    mc = pm.MCMC(vars, db='pickle', dbname='flea')
    mc.use_step_method(pm.AdaptiveMetropolis, [beta, fraction, sigma, i_r0])
    mc.sample(iter=5, burn=1, thin=1, verbose=0)
    mc.db.close()
    # mc.summary()
    M = pm.MAP(mc)
    M.fit()
    M.BIC
    plot(mc)