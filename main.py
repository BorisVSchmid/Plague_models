from models import simple_rat_flea, rat_flea_model

def run_simple(title):
    simple_rat_flea.run(title)

def run_pymc():
    rat_flea_model.run()

if __name__ == "__main__":
    # run_simple("Infection, half resistant rats at t=0")
    run_pymc()


