import time
from datetime import datetime


class Timing:
    
    def __init__(self):
        self.start = time.time()
        self.sample_start = 0
        self.start_date = datetime.now()
        self.duration = 0
        self.days = 0
        self.hours = 0
        self.minutes = 0
        self.seconds = 0
        
    def stop(self):
        self.duration = time.time() - self.start
        self.seconds = self.duration % 60
        self.minutes = self.duration // 60 % 60
        self.hours = self.duration // 60 // 60 % 24
        self.days = self.duration // 60 // 60 // 24

    def started(self):
        return "---- started on {} ----\n".format(self.start_date.strftime("%H:%M %A %d %B %Y"))

    def sample(self):
        self.sample_start = time.time()
        
    def time_str(self, duration):
        seconds = duration % 60
        minutes = duration // 60 % 60
        hours = duration // 60 // 60 % 24
        days = duration // 60 // 60 // 24
        return "{}d:{}h:{}m:{}s".format(days, hours, minutes, round(seconds, 3))

    def project(self, iterations, projected):
        return "---- projected runtime for {} iterations: {} ----"\
            .format(str(projected), self.time_str(((time.time() - self.sample_start) / iterations) * projected))

    def __str__(self):
        return "---- finished in {} ----".format(self.time_str(time.time() - self.start))
