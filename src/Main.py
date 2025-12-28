from src.ArenaSettings import HyperParameters
from src.NNA.engine.NeuroEngine import NeuroEngine


def main():
    shared_hyper = HyperParameters()
    neuro_engine = NeuroEngine(shared_hyper)
    print("end")


if __name__ == '__main__':
    main() #Normal run