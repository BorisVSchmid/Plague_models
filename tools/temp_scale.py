import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.TempReader import TempReader


def plot(temps_yearly, years):
    title = "Temperature during different\n years in Mahajanga"
    fig, ax = plt.subplots()

    # plot the data
    for label in years:
        data = temps_yearly[label]
        xs = np.arange(len(data) + 1)
        mask = temps_yearly[str(label) + "_mask"]
        ax.plot(xs[mask], data[mask], marker='o', linestyle='None', label=str(label) + " outside SD + 0.5 Celcius")
    ax.fill_between(range(1, len(temps_yearly["average 1980-2010"][0]) + 1), temps_yearly["average 1980-2010"][1],
                    temps_yearly["average 1980-2010"][0], color='red', alpha=0.3,
                    label="temperature within SD 1980-2010")

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
    plt.xlabel('time in days')
    plt.ylabel('temperature in Celsius')
    plt.show()
    plt.close()


if __name__ == "__main__":
    data, temp_list = TempReader().cooked()
    years = [1995, 1996, 1997, 1998, 1999]
    temps_yearly = dict()
    average_yearly = [[] for x in range(1, 366)]
    s = 0
    for i, year in enumerate(range(1980, 2010)):
        s = i + 1
        years_list = pd.date_range(datetime.date(year, 1, 1), datetime.date(year, 12, 31)).tolist()
        temps_list = [data[d.strftime("%Y-%m-%d")][0] for d in years_list]
        day = 1
        end = False
        week = []
        weeks = []
        for n, t in enumerate(temps_list, 0):
            n_week = n//7
            if n % 7 == 0 and n_week < 52 and n != 0:
                av = sum(week) / float(len(week))
                weeks.append(av)
                week = []
            if n_week == 52 and n == len(temps_list) - 1:
                week.append(t)
                av = sum(week) / float(len(week))
                weeks.append(av)
                week = []
            week.append(t)
        temps_yearly[year] = temps_list[0:365]
        [average_yearly[n].append(x) for n, x in enumerate(temps_list[0:365])]
    averages = [sum(x)/len(x) for x in average_yearly]
    sd = [np.sqrt(sum([np.square(m - averages[n]) for m in x]) / len(x)) for n, x in enumerate(average_yearly)]
    upper = [x + sd[n] for n, x in enumerate(averages)]
    lower = [x - sd[n] for n, x in enumerate(averages)]
    for k in years:
        v = temps_yearly[k]
        for n, t in enumerate(v):
            if lower[n] - 0.25 <= t <= upper[n] + 0.25:
                temps_yearly[k][n] = None
        temps_yearly[k] = np.asarray(v).astype(np.double)
        temps_yearly[str(k) + "_mask"] = np.isfinite(temps_yearly[k])
    temps_yearly["average 1980-2010"] = [upper, lower]
    plot(temps_yearly, years)
