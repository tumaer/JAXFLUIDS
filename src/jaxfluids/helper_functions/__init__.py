from typing import Tuple

def roll_tuple(a: Tuple, shift: int) -> Tuple:
    """Roll tuple elements by given number of places.
    Similar to jnp.roll or np.roll but for tuples.

    :param a: _description_
    :type a: Tuple
    :param shift: _description_
    :type shift: int
    :return: _description_
    :rtype: Tuple
    """
    return tuple([a[(i - shift) % len(a)] for i, x in enumerate(a)])
