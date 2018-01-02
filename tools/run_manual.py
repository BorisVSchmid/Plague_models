import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from tools.TempReader import TempReader


class Model:
    def __init__(self, directory, *args):
        # -- Params
        self.dir = directory
        self.years_list = args[0][-2]
        self.months_list = args[0][-1]
        self.rat_pop = args[0][1].value
        self.beta_h = args[0][2].value
        self.temp_scale = args[0][5].value
        self.beta_r = args[0][6].value
        self.inh_res = args[0][11].value
        self.data, self.temp_list = TempReader().cooked()
        self.t = [x for x in range(0, len(self.years_list))]
        self.i_h = np.zeros_like(self.t, dtype=float)
        self.r_h = np.zeros_like(self.t, dtype=float)
        self.s_r = np.zeros_like(self.t, dtype=float)
        self.i_r = np.zeros_like(self.t, dtype=float)
        self.res_r = np.zeros_like(self.t, dtype=float)
        self.d_r = np.zeros_like(self.t, dtype=float)
        self.i_f = np.zeros_like(self.t, dtype=float)
        self.fph = np.zeros_like(self.t, dtype=float)

    def graph(self):
        confirmed_cases = [0, 0, 0, 0, 0, 0, 8, 12, 62, 16, 2, 14, 6, 5, 0, 0, 0, 0, 1, 5, 22, 39, 11, 8, 5, 6, 2, 1, 0, 0, 10, 38, 59,
                           74, 13, 6, 1, 1, 0, 0, 0, 0, 4, 17, 18, 29, 9, 8, 3, 3, 1, 0, 1, 0]
        scaled_cases = [0, 0, 0, 0, 0, 0, 52.0, 78.0, 403.0, 104.0, 13.0, 91.0, 36.0, 30.0, 0.0, 0.0, 0.0, 0.0, 6.0, 30.0, 132.0, 234.0, 66.0, 48.0, 15.0, 18.0, 6.0, 3.0, 0.0, 0.0, 30.0, 114.0, 177.0, 222.0, 39.0, 18.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 12.0, 51.0, 54.0, 87.0, 27.0, 24.0, 24.0, 24.0, 8.0, 0.0, 8.0, 0.0]
        self.plot("infected_humans", "graph of infected humans with\n max posteriori values",
                  infected_humans=self.i_h, confirmed_cases=confirmed_cases, scaled_cases=scaled_cases)
        self.plot("infected_rats", "graph of infected rats with\n max posteriori values",
                  susceptible_rats=self.s_r, infected_rats=self.i_r, resistant_rats=self.res_r)

    def plot(self, filename, title, **kwargs):
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')
        fig, ax = plt.subplots()

        # plot the data
        for label, data in kwargs.items():
            if len(data) == len(self.years_list):
                ax.plot(self.years_list, data, label=" ".join(label.split("_")))
            else:
                ax.plot(self.months_list, data, label=" ".join(label.split("_")))

        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        # set the axis limit
        datemin = min(self.years_list)
        datemax = max(self.years_list) + 1
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
        plt.rc('figure', autolayout=True)
        plt.title(title)
        plt.xlabel('time in months')
        plt.ylabel('number of humans')
        plt.savefig(filename + ".png")

    def plague_model(self):
        # - human
        gamma_h = 0.2
        p_recovery_h = .4
        # - rat
        gamma_r = 0.2
        # .1
        p_recovery_ur = .1
        rep_rate_r = .4 * (1 - 0.234)
        rep_rate_ur = .4
        d_rate_ui = 1 / 365
        # - flea
        d_rate = 0.2
        # 0.2
        g_rate = .0084
        c_cap = 6.
        # human
        n_h = 25000
        self.i_h[0] = 0.
        # rat
        self.s_r[0] = self.rat_pop - 20.
        self.res_r[0] = 0.
        infected_rat_deaths = 0.0
        # flea
        n_d_rate = 0.005
        searching = 3. / self.s_r[0]
        self.fph[0] = 6.0
        # shrews
        i_rpd = 0.00167
        for i, v in enumerate(self.years_list[1:], 1):
            if 189 <= v.timetuple().tm_yday <= 222:
                shrew_transference = i_rpd * self.s_r[i - 1]
            else:
                shrew_transference = 0
            date_string = v.strftime("%Y-%m-%d")
            temp = self.data[date_string][0] * self.temp_scale
            temp_growth_factor = max(0, (temp - 15.0) / 10.0)
            temp_spread_factor = (0.75 - 0.25 * np.tanh(((temp * 9. / 5.) + 32.) - 80.))
            # + rec_r[i - 1]
            n_r = self.s_r[i - 1] + self.i_r[i - 1] + self.res_r[i - 1]
            # natural deaths
            natural_death_unresistant = (self.s_r[i - 1] * d_rate_ui)
            natural_death_resistant = (self.res_r[i - 1] * d_rate_ui)
            natural_death_infected = (self.i_r[i - 1] * d_rate_ui)
            # - Fleas
            new_infectious = infected_rat_deaths * self.fph[i - 1]
            # could be made temperature dependent
            starvation_deaths = d_rate * self.i_f[i - 1]
            # number of fleas that find a human
            force_to_humans = min(self.i_f[i - 1], self.i_f[i - 1] * np.exp(float(-searching * n_r)))
            # number of fleas that find a rat
            force_to_rats = self.i_f[i - 1] - force_to_humans
            force_to_rats = force_to_rats * temp_spread_factor
            force_to_humans = force_to_humans * temp_spread_factor
            self.fph[i] = self.fph[i - 1] + (temp_growth_factor * g_rate * self.fph[i - 1])\
                          - (n_d_rate * (1 + self.fph[i - 1] / c_cap) * self.fph[i - 1])
            # should add dehydration
            self.i_f[i] = max(0.0, self.i_f[i - 1] + new_infectious - starvation_deaths)

            # - Rats
            new_infected_rats = self.beta_r * self.s_r[i - 1] * force_to_rats / n_r
            new_infected_rats = 0 if new_infected_rats < 0 else new_infected_rats
            new_removed_rats = gamma_r * (self.i_r[i - 1] - natural_death_infected)
            new_recovered_rats = p_recovery_ur * new_removed_rats
            new_dead_rats = new_removed_rats - new_recovered_rats
            infected_rat_deaths = new_dead_rats

            # born rats
            pressure = n_r / self.rat_pop
            resistant_born_rats = rep_rate_r * self.res_r[i - 1] * (self.inh_res - pressure)
            unresistant_born_rats = ((rep_rate_r * self.res_r[i - 1] * (1 - self.inh_res))
                                     + (rep_rate_ur * self.s_r[i - 1] * (1 - pressure)))

            # time step values
            self.s_r[i] = min(self.rat_pop, self.s_r[i - 1] + unresistant_born_rats - new_infected_rats
                              - natural_death_unresistant - shrew_transference)
            self.i_r[i] = self.i_r[i - 1] + new_infected_rats - new_removed_rats - natural_death_infected\
                + shrew_transference
            self.res_r[i] = self.res_r[i - 1] + new_recovered_rats + resistant_born_rats - natural_death_resistant
            self.d_r[i] = new_dead_rats + natural_death_unresistant + natural_death_resistant + natural_death_infected

            # - Humans
            s_h = n_h - self.i_h[i - 1] - self.r_h[i - 1]
            new_infected_humans = min(n_h, self.beta_h * s_h * force_to_humans / n_h) + 0.000000000001
            new_removed_humans = gamma_h * self.i_h[i - 1]
            new_recovered_humans = p_recovery_h * new_removed_humans

            # time step values
            self.i_h[i] = self.i_h[i - 1] + new_infected_humans - new_removed_humans
            self.r_h[i] = self.r_h[i - 1] + new_recovered_humans
