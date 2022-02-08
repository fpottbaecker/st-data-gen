import os
import pickle


def pickle_to_file(obj, filename):
    os.makedirs(os.path.basename(filename))
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
