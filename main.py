from models.rat_flea_pymc import *
import models.rat_flea_pymc as rfp
import pymc as pm
from pymc.Matplot import plot
import matplotlib.pyplot as plt
import os
import os.path as op
import matplotlib.dates as mdates
from tools.run_manual import Model
from tools.model_timing import Timing


class Analyze:

    def __init__(self, run=None, timing=None, itrs=1000, burn=500, thin=10, project=100000, verbose=0):
        self.timing = timing
        self.verbose = verbose
        self.iter = itrs
        self.burn = burn
        self.thin = thin
        self.project = project
        self.run = run
        self.vars = [confirmed_cases, rat_pop, beta_h, gamma_h, p_recovery_h, temp_scale, beta_r, gamma_r,
                     p_recovery_ur, rep_rate_r, rep_rate_ur, inh_res, d_rate_ui, d_rate, g_rate, c_cap, sim_data,
                     mortality, mortalitysim, years_list,
                     months_list]
        self.dir = None
        self.mc = None

    def re_run(self, itrs=None, burn=None, thin=None, cont=False):
        self.dir = op.join(op.dirname(op.abspath(__file__)), 'runs', self.run)
        os.chdir(self.dir)
        db = pm.database.pickle.load(self.run + '.pickle')
        self.mc = pm.MCMC(self.vars, db=db)
        if cont:
            self.iter = itrs if itrs else self.iter
            self.burn = burn if burn else self.burn
            self.thin = thin if thin else self.thin
            self.mc.sample(iter=self.iter, burn=self.burn, thin=self.thin, verbose=self.verbose)
        self.remove_string_instances()
        self.plot()
        self.run_manual()

    def new_run(self):
        """returns a new directory to store the current run to."""
        self.dir = op.join(op.dirname(op.abspath(__file__)), 'runs')
        dirs = os.listdir(self.dir)
        if len(dirs):
            l_run = max([int(x[3:]) for x in dirs])
            self.dir = op.join(self.dir, 'run' + str(l_run + 1))
        else:
            self.dir = op.join(self.dir, 'run' + str(1))
        if not op.exists(self.dir):
            os.makedirs(self.dir)
        os.chdir(self.dir)
        self.mc = pm.MCMC(self.vars, db='pickle', dbname=self.dir.split("\\")[-1] + ".pickle")
        self.mc.use_step_method(pm.AdaptiveMetropolis, [rat_pop, beta_h, temp_scale, beta_r, inh_res])
        timing.sample()
        self.mc.sample(iter=self.iter, burn=self.burn, thin=self.thin, verbose=self.verbose)
        if timing and self.project > self.iter:
            print(timing.project(self.iter, self.project))
        # self.mc.summary()
        self.remove_string_instances()
        self.plot()
        self.run_manual()

    def remove_string_instances(self):
        for key in self.mc.__dict__.keys():
            if not isinstance(key, basestring):
                del self.mc.__dict__[key]

    def run_manual(self):
        man = Model(self.dir, self.vars)
        man.plague_model()
        man.graph()

    def plot(self):
        mc_map = pm.MAP(self.mc)
        mc_map.fit(tol=.01)
        # iterlim = 250,
        print("BIC score: {}".format(mc_map.BIC))
        plot(self.mc)
        # set years and months
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')
        fig, ax = plt.subplots()
        # plot the data
        ax.plot(months_list, confirmed_cases, 'o', mec='black', color='black', label='confirmed cases')
        ax.plot(months_list, mortalitysim.stats()['mean'], color='red', linewidth=1, label='BPL (mean)')
        y_min = mortalitysim.stats()['quantiles'][2.5]
        y_max = mortalitysim.stats()['quantiles'][97.5]
        ax.fill_between(months_list, y_min, y_max, color='r', alpha=0.3, label='BPL (95% CI)')
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        # set the axis limit
        datemin = min(months_list) - 1
        datemax = max(months_list) + 1
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
        plt.rc('font', size=12)
        plt.rc('lines', linewidth=2)
        plt.title("Plague model fit to laboratory confirmed cases")
        plt.xlabel('Time in months')
        plt.ylabel('Number of infecteds')
        plt.legend()
        plt.savefig(self.dir.split("\\")[-1] + '_fit.png')


if __name__ == "__main__":
    timing = Timing()
    run = "run44"
    Analyze(run).re_run(itrs=1000000, burn=500000, cont=True)
    # Analyze(itrs=100000, burn=50000).new_run()
    timing.stop()
    print(timing)
