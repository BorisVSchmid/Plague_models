class Vector:

    def __init__(self, suseptible):
        # parameters for the vector flea X. cheopis
        self.suseptible = suseptible
        self.infected = 0
        self.dead = 0

class Human:

    def __init__(self, suseptible):
        # parameters for the human factor in the model
        self.suseptible = suseptible

class Host:

    def __init__(self, suseptible):
        # parameters fot the host in the model
        self.suseptible = suseptible
        self.infected = 0
        self.recovered = 0
        self.resistant = 0
        self.dead = 0
