from os.path import dirname, abspath
import os.path


class TempReader:

    def __init__(self):
        self.parent_dir = dirname(dirname(abspath(__file__)))
        self.fname = os.path.join(self.parent_dir, 'data', 'temp_data.csv')
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
