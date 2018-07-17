import os
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, 0)


def load_obj(name):
    if os.path.exists(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        return None
