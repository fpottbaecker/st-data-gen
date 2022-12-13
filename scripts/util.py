import hashlib
import pickle


def sha256(filename):
    with open(filename, "rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(4096), b""):
            digest.update(block)
        return digest.hexdigest()


def pickle_to_file(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
