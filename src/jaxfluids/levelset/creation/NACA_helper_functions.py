from typing import Dict

import jax
import jax.numpy as jnp

Array = jax.Array

def get_five_digit_airfoil_params() -> Dict[int, Dict[str, Array]]:
    """Helper function which initializes a dict
    containing parameters of five-digit NACA airfoils.

    :return: _description_
    :rtype: Dict[int, Dict[str, Array]]
    """

    five_digit_airfoil_params = {
        0: {
            "r" : jnp.array([ [0.15, 0.01, 0.15, 0.20, 0.25], [0.0580, 0.1260, 0.2025, 0.2900, 0.3910] ]),
            "k1": jnp.array([ [0.15, 0.01, 0.15, 0.20, 0.25], [361.4, 51.64, 15.957, 6.643, 3.23] ])
        },
        1: {
            "r"     : jnp.array([ [0.01, 0.15, 0.20, 0.25], [0.13, 0.217, 0.318, 0.441] ]),
            "k1"    : jnp.array([ [0.01, 0.15, 0.20, 0.25], [51.99, 15.739, 6.52, 3.191] ]),
            "k2_k1" : jnp.array([ [0.01, 0.15, 0.20, 0.25], [0.000764, 0.00677, 0.0303, 0.1355] ])
        }
    }

    return five_digit_airfoil_params

def thickness_distribution(t: Array, x: Array) -> Array:
    """Computes the thickness distribution y_t, i.e.,
    the half thickness, along the chord x.

    :param t: maximum thickness
    :type t: Array
    :param x: chord line
    :type x: Array
    :return: thickness distribution
    :rtype: Array
    """

    y_t = 5 * t * (0.2969 * jnp.sqrt(x) 
        + x * (-0.1260 + x * (-0.3516 + x * (0.2843 - 0.1036 * x)))
    )
    
    return y_t

def five_digit_airfoil(digit_code: str, chord_line: Array) -> Array:
    """_summary_
    
    TODO needs comments

    :param digit_code: _description_
    :type digit_code: str
    :param chord_line: _description_
    :type chord_line: Array
    :return: _description_
    :rtype: Array
    """

    five_digit_airfoil_params = get_five_digit_airfoil_params()

    Cl, p, camber_type = int(digit_code[0]) * 3./20., int(digit_code[1])/20., int(digit_code[2]) # TODO different Cl ?
    assert_string = f"NACA_{digit_code:s} five digit airfoil invalid"
    assert int(digit_code[0]) == 2 and int(digit_code[1]) in [1,2,3,4,5] and int(digit_code[2]) in [0,1], assert_string
    r = jnp.interp(p, five_digit_airfoil_params[camber_type]["r"][0], five_digit_airfoil_params[camber_type]["r"][1])
    k1 = jnp.interp(p, five_digit_airfoil_params[camber_type]["k1"][0], five_digit_airfoil_params[camber_type]["k1"][1])

    if camber_type == 0:
        camber_line = k1/6. * (chord_line**3 - 3*r*chord_line**2 + r**2*(3 - r)*chord_line) * ((chord_line >= 0) & (chord_line < r)) + \
                        k1*r**3/6.*(1 - chord_line) * ((chord_line >= r) & (chord_line <= 1))
        dy_dx = k1/6. * (3*chord_line**2 - 6*r*chord_line + r**2*(3 - r)) * ((chord_line >= 0) & (chord_line < r)) + \
                    -k1*r**3/6. * ((chord_line >= r) & (chord_line <= 1))
    else:
        k2_k1 = jnp.interp(p, five_digit_airfoil_params[1]["k2_k1"][0], five_digit_airfoil_params[1]["k2_k1"][1])
        camber_line = k1/6. * ((chord_line - r)**3 - k2_k1*(1 - r)**3*chord_line - r**3*chord_line + r**3)* ((chord_line >= 0) & (chord_line < r)) + \
                        k1/6. * (k2_k1*(chord_line - r)**3 - k2_k1*(1 - r)**3*chord_line - r**3*chord_line + r**3) * ((chord_line >= r) & (chord_line <= 1))
        dy_dx = k1/6. * (3*(chord_line - r)**2 - k2_k1*(1 - r)**3 - r**3) * ((chord_line >= 0) & (chord_line < r)) + \
                    k1/6. * (3*k2_k1*(chord_line - r)**2 - k2_k1*(1 - r)**3 - r**3) * ((chord_line >= r) & (chord_line <= 1))
    theta = jnp.arctan(dy_dx)

    return camber_line, theta

def four_digit_airfoil(digit_code: str, chord_line: Array) -> Array:

    m, p = int(digit_code[0])/100., int(digit_code[1])/10.
    camber_line = m/p**2 * (2*p*chord_line - chord_line**2) * ((chord_line >= 0) & (chord_line < p)) \
        + m/(1-p)**2 * ((1-2*p) + 2*p*chord_line - chord_line**2) * ((chord_line >= p) & (chord_line <= 1))
    dy_dx = 2*m/p**2 * (p-chord_line) * ((chord_line >= 0) & (chord_line < p)) \
        + 2*m/(1-p)**2 * (p-chord_line) * ((chord_line >= p) & (chord_line <= 1))
    theta = jnp.arctan(dy_dx)

    return camber_line, theta