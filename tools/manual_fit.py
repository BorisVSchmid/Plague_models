import collections
from multiprocessing import Pool, cpu_count
import random as r


class Distribution:

    def __init__(self, name, lower=None, upper=None, step=1., value=None):
        self.lower = float(lower) if lower else lower
        self.upper = float(upper) if upper else upper
        if not self.lower and not self.upper:
            raise Exception("both lower or upper should be used.")
        self.value = float(value) if value else value
        self.best_value = None
        self.step = float(step)
        self.name = str(name)
        self.search = None

    def set_value(self):
        self.value = (self.upper + self.lower) / 2


class RandomWalk:

    def __init__(self):
        self.n_score = None
        self.score = float("-infinity")
        self.cpus = cpu_count()
        self.pool = Pool(processes=self.cpus)

    @staticmethod
    def shape_dist(a, b, c):
        ab = RandomWalk.incline(a, b)
        bc = RandomWalk.incline(b, c)
        ac = RandomWalk.incline(a, c)
        if ab == bc and ab == ac:
            if ab < 0:
                return "slope_down"
            elif ab > 0:
                return "slope_up"
            else:
                return "extend_scan"
        else:
            if ab > bc:
                return "spike"
            else:
                return "dip"

    @staticmethod
    def incline(a, b):
        y = a[1] - b[1]
        x = a[0] - b[0]
        if x != 0:
            return y / x
        else:
            return 0

    @staticmethod
    def get_middle(cpos, opp):
        return (cpos + opp) / 2


class GaussianWalk(RandomWalk):

    def __init__(self, iterations, model, *args):
        super().__init__()
        self.iterations = 2 if iterations <= 1 else int(iterations)
        self.model = model
        self.direction = 'f'
        if not isinstance(self.model, collections.Callable):
            raise TypeError("model needs to be a function")
        try:
            if len(args) < 1:
                raise Exception("no priors given")
            for arg in args:
                isinstance(arg, Distribution)
            self.priors = args
        except TypeError:
            "variable: {} should be of type Distribution".format(arg)
        except Exception as e:
            print(e.args)

    def start(self):
        self.chose_priors()
        self.iterate() if self.iterations else self.best_fit()

    def chose_priors(self):
        for prior in self.priors:
            if not prior.value:
                prior.set_value()

    def iterate(self):
        for i in range(self.iterations):
            values = [p.value for p in self.priors]
            self.n_score = self.model(*values)
            for prior in self.priors:
                prior.best_value = prior.value
            self.scan()
            print([p.value for p in self.priors])

    def scan(self):
        for n, prior in enumerate(self.priors):
            print(self.n_score)
            values_u = [p.value for p in self.priors]
            values_l = [p.value for p in self.priors]
            values_u[n] = RandomWalk.get_middle(values_u[n], prior.upper)
            values_l[n] = RandomWalk.get_middle(values_l[n], prior.lower)
            loglike = self.pool.starmap(self.model, (values_u, values_l))
            upper = [values_u[n], loglike[0]]
            lower = [values_l[n], loglike[1]]
            self.decide(lower, prior, upper)

    def decide(self, lower, prior, upper):
        shape = RandomWalk.shape_dist(lower, [prior.value, self.n_score], upper)
        print(shape)
        if shape == "slope_down":
            if lower[1] > self.n_score:
                prior.best_value = lower[0]
                self.n_score = lower[1]
                self.scan()
            else:
                return
        elif shape == "slope_up":
            if upper[1] > self.n_score:
                prior.best_value = upper[0]
                self.n_score = upper[1]
                self.scan()
            else:
                return
        elif shape == "dip":
            if max([upper[1], lower[1]]) > self.n_score:
                prior.best_value = max([upper[0], lower[0]])
                self.n_score = max([upper[1], lower[1]])
                self.scan()
            else:
                return
        elif shape == "spike":
            # fucks up on dis, infinite halves don't make a whole
            prior.upper = upper[0]
            prior.lower = lower[0]
            prior.value = r.uniform(prior.upper, prior.lower)
        else:
            return
