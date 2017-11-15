from models.rat_flea_pymc import *
import pymc as pm
from pymc.Matplot import plot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    vars = [confirmed_cases, sigma, beta, gamma_h, p_recovery_h, phi, rho, gamma_r, p_recovery_ur, rep_rate_r,
            rep_rate_ur, iota, theta, epsilon, d_rate_ui, d_rate, g_rate, c_cap, sim_data, mortality, mortalitysim]
    mc = pm.MCMC(vars, db='pickle', dbname="rat.pickle")
    mc.use_step_method(pm.AdaptiveMetropolis, [sigma, beta, phi, rho, iota, theta, epsilon])
    mc.sample(iter=100000, burn=50000, thin=10, verbose=1)
    mc.summary()
    # load db back
    # db = pm.database.pickle.load('rat.pickle')
    # mc = pm.MCMC(vars, db=db)
    for key in mc.__dict__.keys():
        if not isinstance(key, basestring):
            del mc.__dict__[key]
    M = pm.MAP(mc)
    M.fit(iterlim=250, tol=.01)
    M.BIC
    plot(mc)
    plt.figure(figsize=(10, 10))
    plt.title("Plague Mahajanga")
    plt.xlabel('Day')
    plt.ylabel('Infecteds')
    plt.plot(confirmed_cases, 'o', mec='black', color='black', label='confirmed cases')
    plt.plot(mortalitysim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
    y_min = mortalitysim.stats()['quantiles'][2.5]
    y_max = mortalitysim.stats()['quantiles'][97.5]
    plt.fill_between(range(0, len(confirmed_cases)), y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
    plt.legend()
    plt.show()
