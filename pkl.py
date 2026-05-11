import pickle
from pprint import pprint


def load_pkl(path):
    """
    Load and print contents of a .pkl file.
    """

    with open(path, "rb") as file:
        data = pickle.load(file)

    print("\nLoaded object type:")
    print(type(data))

    print("\nContent:")
    pprint(data)

    return data


if __name__ == "__main__":
    # Example:
    # path =  "tau_local.pkl"
    path = r"data\value_comparison.pkl"

    data = load_pkl(path)