import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pymc3 as pm
from models.factors import Vector, Human, Host


class Model:

    def __init__(self):
        self.human_pop
        self.sus_frac = pm.Uniform('sus_frac', 1e-9, 1., value=.08)
        self.sus_rats = self.human_pop * self.sus_frac
        self.beta_h = pm.Uniform('beta_h', 1e-9, .2, value=.2)
        self.beta_r = pm.Uniform('beta_r', 1e-9, 1.0, value = .08)
        self.infected_rats = pm.Uniform('infected_rats', 0.15, 1.)
        self.vector = Vector()
        self.human = Human(self.human_pop, self.beta_h)
        self.host = Host(self.sus_rats, self.beta_r, self.infected_rats)

    def start(self, t):
        pass

def run():
    pass