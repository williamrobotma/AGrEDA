"""Miscellaneous utility functions."""
import numpy as np


def check_integer_rng(rng):
    """Converts inputs into integer callback rng.

    Checks if `rng` is an seed, or a `np.random.Generator` or
    `np.random.RandomState` object, and return the integers method of an rng.

    Args:
        rng: Random number generator to check. If a seed, will be used to
            instantiate a `np.random.Generator` object.

    Returns:
        Integer callback for rng.
    """
    # Check if np.random.Generator was passed in
    try:
        rng_integers = rng.integers
    except AttributeError:
        pass

    # Check if np.random.RandomState was passed in
    try:
        rng_integers = rng.randint
    except AttributeError:
        pass

    # Check if seed was passed in
    rng_integers = np.random.default_rng(rng).integers

    return rng_integers
