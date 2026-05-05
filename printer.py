import numpy as np

def fmt_value(x, digits=3):
    """
    Formats a scalar or vector value to a string with specified number of digits.
    """
    x = np.asarray(x)

    if x.ndim == 0:
        return f"{float(x):.{digits}f}"

    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in x) + "]"


def print_feature_table(title, values, feature_names=None, digits=3):
    """
    Prints a table of feature values for each state.
    Numerates Features from 0 to F_card-1 if no feature names are provided.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for state, feature_values in values.items():
        print(f"\nState {state}")
        print("-" * 80)
        print(f"{'Feature':<12} {'Value':>20}")
        print("-" * 80)

        for i, value in enumerate(feature_values):
            name = feature_names[i] if feature_names else f"Feature {i}"
            print(f"{name:<12} {fmt_value(value, digits):>20}")