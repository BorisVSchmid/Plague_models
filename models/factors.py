import math


class Vector:

    def __init__(self, gamma=5.2, death_rate=5.0, growth_rate=0.0084, carrying_cap=6.0, free_vectors=0,
                 vectors_per_host=0):
        # parameters for the vector flea X. cheopis
        self.free_vectors = free_vectors
        self.vectors_per_host = vectors_per_host
        self.growth_rate = growth_rate
        self.carrying_cap = carrying_cap
        self.gamma = gamma
        self.death_rate = death_rate

    def new_AVPH(self):
        return self.growth_rate * self.vectors_per_host * \
               (1 - (self.vectors_per_host/self.carrying_cap))

    def new_free_vectors(self, i):
        return ((1 - self.growth_rate) * self.gamma * i * self.vectors_per_host) - \
               (self.death_rate * self.free_vectors)


class Human:

    def __init__(self, susceptible, beta, infected=0.0, recovered=0.0, dead=1., gamma=10.0, prob_of_recovery=0.4):
        # parameters for the human factor in the model
        self.susceptible = susceptible
        self.infected = infected
        self.recovered = recovered
        self.dead = dead
        self.prob_of_recovery = prob_of_recovery
        self.beta = beta
        self.gamma = gamma

    def new_infected(self, f, n):
        return self.beta * \
               ((self.susceptible * f)/n) * \
               (1 - math.exp(-(3.0/self.susceptible)*n)) * \
               (-self.gamma*self.infected)

    def new_recovered(self):
        return self.prob_of_recovery * \
               self.gamma * \
               self.infected

    def new_dead(self):
        return (1 - self.prob_of_recovery) * \
               self.gamma * \
               self.infected


class Host:

    def __init__(self, susceptible, infected, beta, recovered=0, resistant=0, dead=0, gamma=5.0, prob_of_recovery=0.1):
        # parameters fot the host in the model
        self.susceptible = susceptible
        self.infected = infected
        self.recovered = recovered
        self.resistant = resistant
        self.dead = dead
        self.beta = beta
        self.gamma = gamma
        self.prob_of_recovery = prob_of_recovery
        self.n = self.susceptible + self.infected + self.recovered

    def new_susceptible(self, f):
        return -1 * self.beta * \
               ((self.susceptible * f)/self.n) * \
               (1 - math.exp(-(3.0/self.susceptible)*self.n))

    def new_infected(self, f):
        return self.beta * \
               ((self.susceptible * f)/self.n) * \
               (1 - math.exp(-(3.0/self.susceptible)*self.n)) * \
               (-self.gamma*self.infected)

    def new_recovered(self):
        return self.prob_of_recovery * \
               self.gamma * \
               self.infected

    def new_dead(self):
        return (1 - self.prob_of_recovery) * \
               self.gamma * \
               self.infected
