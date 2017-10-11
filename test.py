from models.an_model import *
import pymc as pm
import matplotlib.pyplot as plt
from pymc.Matplot import plot


if __name__ == "__main__":
    vars = [beta_r, gamma, sigma, searching, H_f0, I_r0, R_r0, D_r0, S_h0,
            I_h0, D_h0, D_h, beta_h, pop_size, sus_frac, mortality,
            mortality_data, mortality_sim]
    mc = pm.MCMC(vars, db='pickle', dbname='rat')
    mc.use_step_method(pm.AdaptiveMetropolis, [beta_r, beta_h, sus_frac, I_r0])
    mc.sample(iter=10, burn=5, thin=5, verbose=1)
    mc.db.close()
    mc.summary()
    mc.DIC

    plot(mc)
    plt.figure(figsize=(10, 10))
    plt.title('Barcelona 1490')
    plt.xlabel('Day')
    plt.ylabel('Deaths')
    plt.plot(mortality_data, 'o', mec='black', color='black', label='Observed deaths')
    plt.plot(mortality_sim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
    y_min = mortality_sim.stats()['quantiles'][2.5]
    y_max = mortality_sim.stats()['quantiles'][97.5]
    plt.fill_between(range(0, len(mortality_data)), y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
    plt.legend()
