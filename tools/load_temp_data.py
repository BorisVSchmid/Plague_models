"""This program is designed to read temperature data from a csv file"""
import pandas as pd
from datetime import date
import netCDF4 as nc
from os.path import dirname, abspath


class TempReader:

    def __init__(self):
        self.fname = "C:\\Users\\rolfve\\PycharmProjects\\Plague_models\\data\\temp_data.csv"
        self.temp_dict = {}
        self.temp_list = []
        
    def cooked(self):
        with open(self.fname, "r") as file:
            for line in file.readlines():
                date, data = line.split(':')
                data = list(map(float, data.split(',')))
                self.temp_dict[date] = data
                self.temp_list.append(data[0])
        return self.temp_dict, self.temp_list

class TempLoader:

    def __init__(self, start=1991, end=1999, update=False, floc="PycharmProjects\\Plague_models\\", fname="1107903"):
        self.loc = "C:\\Users\\rolfve\\" + floc + "\\" + fname + ".csv"
        self.parent_dir = dirname(dirname(abspath(__file__)))
        self.start = start
        self.end = end
        self.update = bool(update)
        # years_list = pd.date_range(date(1990, 1, 1), date(2000, 12, 31)).tolist()
        self.years_range = pd.date_range(date(self.start, 1, 1), date(self.end, 12, 31)).tolist()
        # -- Params
        self.temp = [[27.1, 82], [27.2, 83], [27.5, 81], [27.4, 73], [26.1, 67], [24.6, 64], [24.2, 62], [24.6, 60],
                     [25.4, 63], [26.8, 66], [27.7, 72], [27.4, 80]]
        self.warming = {"1980": 0.150, "1981": 0.145, "1982": 0.19, "1983": 0.19, "1984": 0.165, "1985": 0.140,
                        "1986": 0.175, "1987": 0.2, "1988": 0.24, "1989": 0.275, "1990": 0.31, "1991": 0.30,
                        "1992": 0.28, "1993": 0.285, "1994": 0.29, "1995": 0.285, "1996": 0.318, "1997": 0.351,
                        "1998": 0.384, "1999": 0.417, "2000": 0.45, "2001": 0.483, "2002": 0.516, "2003": 0.549,
                        "2004": 0.582, "2005": 0.590, "2006": 0.582, "2007": 0.575, "2008": 0.590, "2009": 0.582,
                        "2010": 0.575}
        self.template_temps = {}
        self.temp_missing = []
        self.temp_list = []
        for time in self.years_range:
            time = time.strftime("%Y-%m-%d")
            time_s = time.split("-")
            year = time_s[0]
            self.template_temps[time] = list(map(lambda x: x + self.warming[year], self.temp[int(time_s[1]) - 1][:-1]))
            self.template_temps[time].append(self.temp[int(time_s[1]) - 1][1])

    def read_raw(self):
        with open(self.loc, 'r') as file:
            count = 0
            prev_day = 0
            prev_temp = 0
            for line in file.readlines():
                if count > 0:
                    line = line.replace("\n", '')
                    line = line[1:-1]
                    data = line.split('\",\"')
                    year, month, day = data[2].split("-")
                    year = int(year)
                    month = int(month)
                    day = int(day)
                    temp = float(data[3])
                    if year >= self.start and year <= self.end:
                        if day != prev_day and day - prev_day > 1:
                            span = day - prev_day - 1
                            if span == 1:
                                s_temp = (temp + prev_temp) / 2
                                self.template_temps["{}-{}-{}".format(year, TempLoader.str_int(month), TempLoader.str_int(prev_day + 1))][0] = round(s_temp, 2)
                            else:
                                if temp != prev_temp:
                                    s_temp = (temp - prev_temp) / span
                                    for i in range(1, span):
                                        i_temp = prev_temp + (i * s_temp)
                                        date = "{}-{}-{}".format(year, TempLoader.str_int(month),
                                                                 TempLoader.str_int(prev_day + i))
                                        self.template_temps[date][0] = round(i_temp, 2)
                                else:
                                    for i in range(i, span):
                                        date = "{}-{}-{}".format(year, TempLoader.str_int(month),
                                                                 TempLoader.str_int(prev_day + i))
                                        self.template_temps[date][0] = temp
                        self.template_temps[data[2]][0] = temp
                        prev_temp = temp
                        prev_day = day
                count += 1
        self.fill_holes()
        if self.update:
            self.update_data()
        else:
            for n, i in enumerate(self.years_range):
                if n >= 273 and n <= 396:
                    self.template_temps[i.strftime("%Y-%m-%d")][0] = self.temp_missing[n - 273]
                self.temp_list.append(self.template_temps[i.strftime("%Y-%m-%d")][0])
        return self.template_temps, self.temp_list

    def update_data(self):
        with open(self.parent_dir + "\\data\\temp_data.csv", 'w') as file:
            for n, i in enumerate(self.years_range):
                offset = (1991 - self.start)*365
                if n >= offset + 273 and n <= offset + 396:
                    self.template_temps[i.strftime("%Y-%m-%d")][0] = self.temp_missing[n - (offset + 273)]
                file.write(i.strftime("%Y-%m-%d") + ":{},{}".format(self.template_temps[i.strftime("%Y-%m-%d")][0],
                                                                    self.template_temps[i.strftime("%Y-%m-%d")][1]) + "\n")
                self.temp_list.append(self.template_temps[i.strftime("%Y-%m-%d")][0])

    def fill_holes(self):
        tmax_1991_data = nc.Dataset(self.parent_dir + '\\data\\tmax.1991.nc', 'r')
        tmin_1991_data = nc.Dataset(self.parent_dir + '\\data\\tmin.1991.nc', 'r')
        tmax_1992_data = nc.Dataset(self.parent_dir + '\\data\\tmax.1992.nc', 'r')
        tmin_1992_data = nc.Dataset(self.parent_dir + '\\data\\tmin.1992.nc', 'r')
        tmaxs_1991 = tmax_1991_data.variables['tmax'][273:]
        tmins_1991 = tmin_1991_data.variables['tmin'][273:]
        tmaxs_1992 = tmax_1992_data.variables['tmax'][:32]
        tmins_1992 = tmin_1992_data.variables['tmin'][:32]
        for i in range(len(tmaxs_1991)):
            combo = []
            for la in [210, 211, 212]:
                for lo in [92, 93, 94]:
                    combo.append((tmaxs_1991[i][la][lo] + tmins_1991[i][la][lo]) / 2)
            self.temp_missing.append(round(sum(combo) / float(len(combo)), 2))
        tmax_1991_data.close()
        tmin_1991_data.close()
        for i in range(len(tmaxs_1992)):
            combo = []
            for la in [210, 211, 212]:
                for lo in [92, 93, 94]:
                    combo.append((tmaxs_1992[i][la][lo] + tmins_1992[i][la][lo]) / 2)
            self.temp_missing.append(round(sum(combo) / float(len(combo)), 2))
        tmax_1992_data.close()
        tmin_1992_data.close()

    @staticmethod
    def str_int(integer):
        if integer > 9:
            return str(integer)
        else:
            return "0{}".format(integer)
