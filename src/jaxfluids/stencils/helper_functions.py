from typing import List, Tuple

import jax.numpy as jnp
from jax import Array

def compute_coefficients_stretched_mesh_muscl3(
        is_mesh_stretching: List,
        cell_sizes: List,
        slices_mesh: List,
        slices_cell_sizes: List
    ) -> Tuple[List, List, List]:
    """Computes coefficients for the MUSCL3 cell-face reconstruction.
    Specifically, we compute coefficients for the central/upwind differences ratio
    and the upwind polynomial.

    :param is_mesh_stretching: _description_
    :type is_mesh_stretching: List
    :param cell_sizes: _description_
    :type cell_sizes: List
    :return: _description_
    :rtype: Tuple[List, List, List]
    """
    # TODO: refactor such that return values are jax arrays,
    # with simpler slicing for device_id

    c_upwind_ = [[], []]
    c_ratio_ = [[], []]
    for j in [0, 1]:
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                cell_sizes_i = cell_sizes_i[slices_cell_sizes[i]]

                s0_ = slices_mesh[j][i][0]
                s1_ = slices_mesh[j][i][1]
                s2_ = slices_mesh[j][i][2]

                delta_x0 = cell_sizes_i[s0_] # x_{i-1}
                delta_x1 = cell_sizes_i[s1_] # x_{i}
                delta_x2 = cell_sizes_i[s2_] # x_{i+1}

                # Ratio of central/upwind mesh
                c_upwind = delta_x1 / (delta_x0 + delta_x1)               
                c_ratio = (delta_x0 + delta_x1) / (delta_x1 + delta_x2) 

            else:
                c_upwind = 0.5
                c_ratio = 1.0

            c_upwind_[j].append(jnp.array(c_upwind))
            c_ratio_[j].append(jnp.array(c_ratio))

    return c_upwind_, c_ratio_

def compute_coefficients_stretched_mesh_weno3(
        cr_uniform: List,
        betar_uniform: List,
        dr_uniform: List,
        is_mesh_stretching: List,
        cell_sizes: List,
        slices_mesh: List,
        slices_cell_sizes: List
    ) -> Tuple[List, List, List]:
    """Computes coefficients for the WENO3 cell-face reconstruction 
    polynomials, the smoothness measures, and the ideal weights for 
    a uniform or stretched grid.

    :param is_mesh_stretching: _description_
    :type is_mesh_stretching: List
    :param cell_sizes: _description_
    :type cell_sizes: List
    :return: _description_
    :rtype: Tuple[List, List, List]
    """
    # TODO: refactor such that return values are jax arrays,
    # with simpler slicing for device_id

    cr_ = [[], []]
    betar_ = [[], []]
    dr_ = [[], []]
    for j in [0, 1]:
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                cell_sizes_i = cell_sizes_i[slices_cell_sizes[i]]

                s0_ = slices_mesh[j][i][0]
                s1_ = slices_mesh[j][i][1]
                s2_ = slices_mesh[j][i][2]

                delta_x0 = cell_sizes_i[s0_] # x_{i-1}
                delta_x1 = cell_sizes_i[s1_] # x_{i}
                delta_x2 = cell_sizes_i[s2_] # x_{i+1}

                # Polynomial coefficients
                c0_0 = -delta_x1 / (delta_x0 + delta_x1)
                c0_1 = (2 * delta_x1 + delta_x0) / (delta_x0 + delta_x1)

                c1_0 = delta_x2 / (delta_x1 + delta_x2)
                c1_1 = delta_x1 / (delta_x1 + delta_x2)

                # Smoothness coefficients
                beta0_0 = -2.0 * delta_x1 / (delta_x0 + delta_x1)
                beta0_1 = 2.0 * delta_x1 / (delta_x0 + delta_x1)

                beta1_0 = -2.0 * delta_x1 / (delta_x1 + delta_x2)
                beta1_1 = 2.0 * delta_x1 / (delta_x1 + delta_x2)

                # Ideal weights
                d0 = delta_x2 / (delta_x0 + delta_x1 + delta_x2)
                d1 = (delta_x0 + delta_x1) / (delta_x0 + delta_x1 + delta_x2)

            else:
                c0_0, c0_1 = cr_uniform[0]
                c1_0, c1_1 = cr_uniform[1]

                beta0_0, beta0_1 = betar_uniform[0]
                beta1_0, beta1_1 = betar_uniform[1]

                d0, d1 = dr_uniform

            cr_[j].append([
                [jnp.array(c0_0), jnp.array(c0_1)],
                [jnp.array(c1_0), jnp.array(c1_1)]
            ])

            betar_[j].append([
                [jnp.array(beta0_0), jnp.array(beta0_1)],
                [jnp.array(beta1_0), jnp.array(beta1_1)]
            ])

            dr_[j].append([jnp.array(d0), jnp.array(d1)])

    return cr_, betar_, dr_

def compute_coefficients_stretched_mesh_weno5(
        cr_uniform: List,
        betar_uniform: List,
        dr_uniform: List,
        is_mesh_stretching: List,
        cell_sizes: List,
        slices_mesh: List,
        slices_cell_sizes: List
    ) -> Tuple[List, List, List]:
    """Computes coefficients for the WENO5 cell-face reconstruction
    polynomials, the smoothness measures, and the ideal weights for
    a uniform or stretched grid.

    :param is_mesh_stretching: _description_
    :type is_mesh_stretching: List
    :param cell_sizes: _description_
    :type cell_sizes: List
    :return: _description_
    :rtype: Tuple[List, List, List]
    """

    cr_ = [[], []]
    betar_ = [[], []]
    dr_ = [[], []]
    for j in [0, 1]:
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                
                cell_sizes_i = cell_sizes_i[slices_cell_sizes[i]]

                s0_ = slices_mesh[j][i][0]
                s1_ = slices_mesh[j][i][1]
                s2_ = slices_mesh[j][i][2]
                s3_ = slices_mesh[j][i][3]
                s4_ = slices_mesh[j][i][4]

                delta_x0 = cell_sizes_i[s0_] # x_{i-2}
                delta_x1 = cell_sizes_i[s1_] # x_{i-1}
                delta_x2 = cell_sizes_i[s2_] # x_{i}
                delta_x3 = cell_sizes_i[s3_] # x_{i+1}
                delta_x4 = cell_sizes_i[s4_] # x_{i+2}

                # POLYNOMIALS P_03, P_13, P_23
                one_c0 = 1.0 / (delta_x0 + delta_x1) / (delta_x1 + delta_x2) / (delta_x0 + delta_x1 + delta_x2) 
                c0_0 = one_c0 * delta_x2 * (
                    (delta_x2 + delta_x1) * (delta_x2 + delta_x1))
                c0_1 = one_c0 * (-delta_x2) * (
                    1 * delta_x2 * delta_x2 + \
                    4 * delta_x2 * delta_x1 + \
                    2 * delta_x2 * delta_x0 + \
                    3 * delta_x1 * delta_x1 + \
                    3 * delta_x1 * delta_x0 + \
                    1 * delta_x0 * delta_x0)
                c0_2 = one_c0 * (delta_x1 + delta_x0) * (
                    3 * delta_x2 * delta_x2 + \
                    4 * delta_x2 * delta_x1 + \
                    2 * delta_x2 * delta_x0 + \
                    1 * delta_x1 * delta_x1 + \
                    1 * delta_x1 * delta_x0)

                one_c1 = 1.0 / (delta_x1 + delta_x2) / (delta_x2 + delta_x3) / (delta_x1 + delta_x2 + delta_x3) 
                c1_0 = one_c1 * (-delta_x2) * delta_x3 * (delta_x2 + delta_x3)
                c1_1 = one_c1 * delta_x3 * (
                    3 * delta_x2 * delta_x2 + \
                    2 * delta_x2 * delta_x3 + \
                    3 * delta_x2 * delta_x1 + \
                    1 * delta_x3 * delta_x1 + \
                    1 * delta_x1 * delta_x1)
                c1_2 = one_c1 * delta_x2 * (delta_x2 + delta_x1) * (delta_x2 + delta_x1)

                one_c2 = 1.0 / (delta_x2 + delta_x3) / (delta_x3 + delta_x4) / (delta_x2 + delta_x3 + delta_x4) 
                c2_0 = one_c2 * delta_x3 * (delta_x3 + delta_x4) * (delta_x3 + delta_x4)
                c2_1 = one_c2 * delta_x2 * (
                    2 * delta_x2 * delta_x3 + \
                    1 * delta_x2 * delta_x4 + \
                    3 * delta_x3 * delta_x3 + \
                    3 * delta_x3 * delta_x4 + \
                    1 * delta_x4 * delta_x4)
                c2_2 = one_c2 * (-delta_x2) * delta_x3 * (delta_x2 + delta_x3)

                # SMOOTHNESS MEASURES BETA_03, BETA_13, BETA_23
                (beta0_0, beta0_1, beta0_2, beta0_3, beta0_4, beta0_5), \
                (beta1_0, beta1_1, beta1_2, beta1_3, beta1_4, beta1_5), \
                (beta2_0, beta2_1, beta2_2, beta2_3, beta2_4, beta2_5), \
                _ = compute_smothnesses3(delta_x0, delta_x1, delta_x2, delta_x3, delta_x4, 0.0)

                # IDEAL WEIGHTS D_0, D_1, D_2
                d0 = delta_x3 * (delta_x3 + delta_x4) / (
                    (delta_x2 + delta_x3 + delta_x1 + delta_x0) * \
                    (delta_x2 + delta_x3 + delta_x4 + delta_x1 + delta_x0)
                ) 
                d1 = (delta_x3 + delta_x4) * (delta_x2 + delta_x1 + delta_x0) * (2 * delta_x2 + 2 * delta_x3 + delta_x4 + 2 * delta_x1 + delta_x0) / (
                    (delta_x2 + delta_x3 + delta_x4 + delta_x1) * \
                    (delta_x2 + delta_x3 + delta_x1 + delta_x0) * \
                    (delta_x2 + delta_x3 + delta_x4 + delta_x1 + delta_x0)
                )
                d2 = (delta_x2 + delta_x1) * (delta_x2 + delta_x1 + delta_x0) / (
                    (delta_x2 + delta_x3 + delta_x4 + delta_x1) * \
                    (delta_x2 + delta_x3 + delta_x4 + delta_x1 + delta_x0)
                )

            else:
                c0_0, c0_1, c0_2 = cr_uniform[0]
                c1_0, c1_1, c1_2 = cr_uniform[1]
                c2_0, c2_1, c2_2 = cr_uniform[2]

                beta0_0, beta0_1, beta0_2, beta0_3, beta0_4, beta0_5 = betar_uniform[0]
                beta1_0, beta1_1, beta1_2, beta1_3, beta1_4, beta1_5 = betar_uniform[1]
                beta2_0, beta2_1, beta2_2, beta2_3, beta2_4, beta2_5 = betar_uniform[2]

                d0, d1, d2 = dr_uniform

            cr_[j].append([
                [jnp.array(c0_0), jnp.array(c0_1), jnp.array(c0_2)],
                [jnp.array(c1_0), jnp.array(c1_1), jnp.array(c1_2)],
                [jnp.array(c2_0), jnp.array(c2_1), jnp.array(c2_2)],
            ])

            betar_[j].append([
                [jnp.array(beta0_0), jnp.array(beta0_1), jnp.array(beta0_2),
                 jnp.array(beta0_3), jnp.array(beta0_4), jnp.array(beta0_5)],
                [jnp.array(beta1_0), jnp.array(beta1_1), jnp.array(beta1_2),
                 jnp.array(beta1_3), jnp.array(beta1_4), jnp.array(beta1_5)],
                [jnp.array(beta2_0), jnp.array(beta2_1), jnp.array(beta2_2),
                 jnp.array(beta2_3), jnp.array(beta2_4), jnp.array(beta2_5)],
            ])

            dr_[j].append([jnp.array(d0), jnp.array(d1), jnp.array(d2)])

    return cr_, betar_, dr_

def compute_coefficients_stretched_mesh_weno6(
        cr_uniform: List,
        betar_uniform: List,
        dr_uniform: List,
        is_mesh_stretching: List,
        cell_sizes: List,
        slices_mesh: List,
        slices_cell_sizes: List
    ) -> Tuple[List, List, List]:
    """Computes coefficients for the WENO6 cell-face reconstruction 
    polynomials, the smoothness measures, and the ideal weights for 
    a uniform or stretched grid.

    :param is_mesh_stretching: _description_
    :type is_mesh_stretching: List
    :param cell_sizes: _description_
    :type cell_sizes: List
    :return: _description_
    :rtype: Tuple[List, List, List]
    """

    cr_ = [[], []]
    betar_ = [[], []]
    dr_ = [[], []]
    cicj_ = [[], []]
    ci_ = [[], []]
    for j in [0, 1]:
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                
                cell_sizes_i = cell_sizes_i[slices_cell_sizes[i]]

                s0_ = slices_mesh[j][i][0]
                s1_ = slices_mesh[j][i][1]
                s2_ = slices_mesh[j][i][2]
                s3_ = slices_mesh[j][i][3]
                s4_ = slices_mesh[j][i][4]
                s5_ = slices_mesh[j][i][5]

                delta_x0 = delta_m2 = cell_sizes_i[s0_] # x_{i-2}
                delta_x1 = delta_m1 = cell_sizes_i[s1_] # x_{i-1}
                delta_x2 = delta_m0 = cell_sizes_i[s2_] # x_{i}
                delta_x3 = delta_p1 = cell_sizes_i[s3_] # x_{i+1}
                delta_x4 = delta_p2 = cell_sizes_i[s4_] # x_{i+2}
                delta_x5 = delta_p3 = cell_sizes_i[s5_] # x_{i+3}

                (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
                (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
                (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
                = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # POLYNOMIALS P_03, P_13, P_23, P_33
                (c0_0, c0_1, c0_2), (c1_0, c1_1, c1_2), \
                (c2_0, c2_1, c2_2), (c3_0, c3_1, c3_2) \
                = compute_polynomials3(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # P_6
                c6_0, c6_1, c6_2, c6_3, c6_4, c6_5 \
                = compute_polynomial6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # SMOOTHNESS MEASURES BETA_03, BETA_13, BETA_23, BETA_33, BETA_6
                # BETA_03, BETA_13, BETA_23
                (beta0_0, beta0_1, beta0_2, beta0_3, beta0_4, beta0_5), \
                (beta1_0, beta1_1, beta1_2, beta1_3, beta1_4, beta1_5), \
                (beta2_0, beta2_1, beta2_2, beta2_3, beta2_4, beta2_5), \
                (beta3_0, beta3_1, beta3_2, beta3_3, beta3_4, beta3_5) \
                = compute_smothnesses3(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # BETA_6
                (C00, C10, C11, C20, C21, C22, C30, C31, C32, C33, C40, C41, C42, C43, C44,
                C50, C51, C52, C53, C54, C55), \
                (coeff_C0C0, coeff_C0C1, coeff_C0C2, coeff_C0C3, coeff_C0C4, coeff_C0C5,
                coeff_C1C1, coeff_C1C2, coeff_C1C3, coeff_C1C4, coeff_C1C5,
                coeff_C2C2, coeff_C2C3, coeff_C2C4, coeff_C2C5,
                coeff_C3C3, coeff_C3C4, coeff_C3C5,
                coeff_C4C4, coeff_C4C5,
                coeff_C5C5) = compute_smoothness_beta6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # IDEAL WEIGHTS D_0, D_1, D_2, D_3
                d0 = c6_0 / c0_0
                d1 = (c6_1 - c0_1 * d0) / c1_0
                d3 = c6_5 / c3_2
                d2 = (c6_4 - c3_1 * d3) / c2_2

            else:
                c0_0, c0_1, c0_2 = cr_uniform[0]
                c1_0, c1_1, c1_2 = cr_uniform[1]
                c2_0, c2_1, c2_2 = cr_uniform[2]
                c3_0, c3_1, c3_2 = cr_uniform[3]

                beta0_0, beta0_1, beta0_2, beta0_3, beta0_4, beta0_5 = betar_uniform[0]
                beta1_0, beta1_1, beta1_2, beta1_3, beta1_4, beta1_5 = betar_uniform[1]
                beta2_0, beta2_1, beta2_2, beta2_3, beta2_4, beta2_5 = betar_uniform[2]
                beta3_0, beta3_1, beta3_2, beta3_3, beta3_4, beta3_5 = betar_uniform[3]

                C00 = C10 = C11 = C20 = C21 = C22 = C30 = C31 = C32 = C33 \
                = C40 = C41 = C42 = C43 = C44 = C50 = C51 = C52 = C53 \
                = C54 = C55 = 0

                coeff_C0C0 = coeff_C0C1 = coeff_C0C2 = coeff_C0C3 = coeff_C0C4 = coeff_C0C5 \
                = coeff_C1C1 = coeff_C1C2 = coeff_C1C3 = coeff_C1C4 = coeff_C1C5 \
                = coeff_C2C2 = coeff_C2C3 = coeff_C2C4 = coeff_C2C5 \
                = coeff_C3C3 = coeff_C3C4 = coeff_C3C5 \
                = coeff_C4C4 = coeff_C4C5 \
                = coeff_C5C5 = 0

                d0, d1, d2, d3 = dr_uniform

            cr_[j].append([
                [jnp.array(c0_0), jnp.array(c0_1), jnp.array(c0_2)],
                [jnp.array(c1_0), jnp.array(c1_1), jnp.array(c1_2)],
                [jnp.array(c2_0), jnp.array(c2_1), jnp.array(c2_2)],
                [jnp.array(c3_0), jnp.array(c3_1), jnp.array(c3_2)],
            ])

            betar_[j].append([
                [jnp.array(beta0_0), jnp.array(beta0_1), jnp.array(beta0_2),
                 jnp.array(beta0_3), jnp.array(beta0_4), jnp.array(beta0_5)],
                [jnp.array(beta1_0), jnp.array(beta1_1), jnp.array(beta1_2),
                 jnp.array(beta1_3), jnp.array(beta1_4), jnp.array(beta1_5)],
                [jnp.array(beta2_0), jnp.array(beta2_1), jnp.array(beta2_2),
                 jnp.array(beta2_3), jnp.array(beta2_4), jnp.array(beta2_5)],
                [jnp.array(beta3_0), jnp.array(beta3_1), jnp.array(beta3_2),
                 jnp.array(beta3_3), jnp.array(beta3_4), jnp.array(beta3_5)],
            ])

            ci_[j].append([
                jnp.array(C00),
                jnp.array(C10), jnp.array(C11),
                jnp.array(C20), jnp.array(C21), jnp.array(C22),
                jnp.array(C30), jnp.array(C31), jnp.array(C32), jnp.array(C33),
                jnp.array(C40), jnp.array(C41), jnp.array(C42), jnp.array(C43), jnp.array(C44),
                jnp.array(C50), jnp.array(C51), jnp.array(C52), jnp.array(C53), jnp.array(C54), jnp.array(C55),
            ])

            cicj_[j].append([
                jnp.array(coeff_C0C0), jnp.array(coeff_C0C1), jnp.array(coeff_C0C2), jnp.array(coeff_C0C3), jnp.array(coeff_C0C4), jnp.array(coeff_C0C5),
                jnp.array(coeff_C1C1), jnp.array(coeff_C1C2), jnp.array(coeff_C1C3), jnp.array(coeff_C1C4), jnp.array(coeff_C1C5),
                jnp.array(coeff_C2C2), jnp.array(coeff_C2C3), jnp.array(coeff_C2C4), jnp.array(coeff_C2C5),
                jnp.array(coeff_C3C3), jnp.array(coeff_C3C4), jnp.array(coeff_C3C5),
                jnp.array(coeff_C4C4), jnp.array(coeff_C4C5),
                jnp.array(coeff_C5C5),
            ])

            dr_[j].append([jnp.array(d0), jnp.array(d1), jnp.array(d2), jnp.array(d3)])

    return cr_, betar_, dr_, ci_, cicj_

def compute_coefficients_stretched_mesh_teno6(
        cr_uniform: List,
        betar_uniform: List,
        dr_uniform: List,
        is_mesh_stretching: List,
        cell_sizes: List,
        slices_mesh: List,
        slices_cell_sizes: List
    ) -> Tuple[List, List, List]:
    """Computes coefficients for the TENO6 cell-face reconstruction 
    polynomials, the smoothness measures, and the ideal weights for 
    a uniform or stretched grid.

    :param is_mesh_stretching: _description_
    :type is_mesh_stretching: List
    :param cell_sizes: _description_
    :type cell_sizes: List
    :return: _description_
    :rtype: Tuple[List, List, List]
    """

    cr_ = [[], []]
    betar_ = [[], []]
    dr_ = [[], []]
    cicj_ = [[], []]
    ci_ = [[], []]
    ci_beta4_ = [[], []]
    cicj_beta4_ = [[], []]
    for j in [0, 1]:
        for i, axis in enumerate(["x", "y", "z"]):
            cell_sizes_i = cell_sizes[i]
            if is_mesh_stretching[i]:
                
                cell_sizes_i = cell_sizes_i[slices_cell_sizes[i]]

                s0_ = slices_mesh[j][i][0]
                s1_ = slices_mesh[j][i][1]
                s2_ = slices_mesh[j][i][2]
                s3_ = slices_mesh[j][i][3]
                s4_ = slices_mesh[j][i][4]
                s5_ = slices_mesh[j][i][5]

                delta_x0 = delta_m2 = cell_sizes_i[s0_] # x_{i-2}
                delta_x1 = delta_m1 = cell_sizes_i[s1_] # x_{i-1}
                delta_x2 = delta_m0 = cell_sizes_i[s2_] # x_{i}
                delta_x3 = delta_p1 = cell_sizes_i[s3_] # x_{i+1}
                delta_x4 = delta_p2 = cell_sizes_i[s4_] # x_{i+2}
                delta_x5 = delta_p3 = cell_sizes_i[s5_] # x_{i+3}

                (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
                (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
                (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
                = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # POLYNOMIALS P_03, P_13, P_23, P_34, P_6
                # P_03, P_13, P_23 
                (c0_0, c0_1, c0_2), (c1_0, c1_1, c1_2), \
                (c2_0, c2_1, c2_2), _ \
                = compute_polynomials3(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)
                
                # P_34
                c3_0, c3_1, c3_2, c3_3 \
                = compute_polynomial34(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # P_6
                c6_0, c6_1, c6_2, c6_3, c6_4, c6_5 \
                = compute_polynomial6(delta_m2, delta_m1, delta_m0, delta_p1, delta_p2, delta_p3)

                # SMOOTHNESS MEASURES BETA_03, BETA_13, BETA_23, BETA_34, BETA_6
                # BETA_03, BETA_13, BETA_23
                (beta0_0, beta0_1, beta0_2, beta0_3, beta0_4, beta0_5), \
                (beta1_0, beta1_1, beta1_2, beta1_3, beta1_4, beta1_5), \
                (beta2_0, beta2_1, beta2_2, beta2_3, beta2_4, beta2_5), \
                _ = \
                compute_smothnesses3(delta_m2, delta_m1, delta_m0, delta_p1, delta_p2, delta_p3)
                
                # BETA_34
                (beta_C00, beta_C10, beta_C11, beta_C20, beta_C21, beta_C22,
                beta_C30, beta_C31, beta_C32, beta_C33), \
                (coeff_beta3_C0C0, coeff_beta3_C0C1, coeff_beta3_C0C2, coeff_beta3_C0C3,
                coeff_beta3_C1C1, coeff_beta3_C1C2, coeff_beta3_C1C3,
                coeff_beta3_C2C2, coeff_beta3_C2C3,
                coeff_beta3_C3C3) = compute_smoothness_beta34(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # BETA_6
                (C00, C10, C11, C20, C21, C22, C30, C31, C32, C33, C40, C41, C42, C43, C44,
                C50, C51, C52, C53, C54, C55), \
                (coeff_C0C0, coeff_C0C1, coeff_C0C2, coeff_C0C3, coeff_C0C4, coeff_C0C5,
                coeff_C1C1, coeff_C1C2, coeff_C1C3, coeff_C1C4, coeff_C1C5,
                coeff_C2C2, coeff_C2C3, coeff_C2C4, coeff_C2C5,
                coeff_C3C3, coeff_C3C4, coeff_C3C5,
                coeff_C4C4, coeff_C4C5,
                coeff_C5C5) = compute_smoothness_beta6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

                # IDEAL WEIGHTS
                d0 = c6_0 / c0_0
                d1 = (c6_1 - c0_1 * d0) / c1_0
                d3 = c6_5 / c3_3
                d2 = (c6_4 - c3_2 * d3) / c2_2

            else:
                c0_0, c0_1, c0_2 = cr_uniform[0]
                c1_0, c1_1, c1_2 = cr_uniform[1]
                c2_0, c2_1, c2_2 = cr_uniform[2]
                c3_0, c3_1, c3_2, c3_3 = cr_uniform[3]

                beta0_0, beta0_1, beta0_2, beta0_3, beta0_4, beta0_5 = betar_uniform[0]
                beta1_0, beta1_1, beta1_2, beta1_3, beta1_4, beta1_5 = betar_uniform[1]
                beta2_0, beta2_1, beta2_2, beta2_3, beta2_4, beta2_5 = betar_uniform[2]
                beta3_0, beta3_1, beta3_2, beta3_3, beta3_4, beta3_5, \
                beta3_6, beta3_7, beta3_8, beta3_9 = betar_uniform[3]

                d0, d1, d2, d3 = dr_uniform

                C00 = C10 = C11 = C20 = C21 = C22 = C30 = C31 = C32 = C33 \
                = C40 = C41 = C42 = C43 = C44 = C50 = C51 = C52 = C53 \
                = C54 = C55 = 0

                coeff_C0C0 = coeff_C0C1 = coeff_C0C2 = coeff_C0C3 = coeff_C0C4 = coeff_C0C5 \
                = coeff_C1C1 = coeff_C1C2 = coeff_C1C3 = coeff_C1C4 = coeff_C1C5 \
                = coeff_C2C2 = coeff_C2C3 = coeff_C2C4 = coeff_C2C5 \
                = coeff_C3C3 = coeff_C3C4 = coeff_C3C5 \
                = coeff_C4C4 = coeff_C4C5 \
                = coeff_C5C5 = 0

                beta_C00 = beta_C10 = beta_C11 \
                = beta_C20 = beta_C21 = beta_C22 \
                = beta_C30 = beta_C31 = beta_C32 = beta_C33 = 0

                coeff_beta3_C0C0 = coeff_beta3_C0C1 = coeff_beta3_C0C2 = coeff_beta3_C0C3 \
                = coeff_beta3_C1C1 = coeff_beta3_C1C2 = coeff_beta3_C1C3 \
                = coeff_beta3_C2C2 = coeff_beta3_C2C3 \
                = coeff_beta3_C3C3 = 0

            cr_[j].append([
                [jnp.array(c0_0), jnp.array(c0_1), jnp.array(c0_2)],
                [jnp.array(c1_0), jnp.array(c1_1), jnp.array(c1_2)],
                [jnp.array(c2_0), jnp.array(c2_1), jnp.array(c2_2)],
                [jnp.array(c3_0), jnp.array(c3_1), jnp.array(c3_2), jnp.array(c3_3)],
            ])

            betar_[j].append([
                [jnp.array(beta0_0), jnp.array(beta0_1), jnp.array(beta0_2),
                 jnp.array(beta0_3), jnp.array(beta0_4), jnp.array(beta0_5)],
                [jnp.array(beta1_0), jnp.array(beta1_1), jnp.array(beta1_2),
                 jnp.array(beta1_3), jnp.array(beta1_4), jnp.array(beta1_5)],
                [jnp.array(beta2_0), jnp.array(beta2_1), jnp.array(beta2_2),
                 jnp.array(beta2_3), jnp.array(beta2_4), jnp.array(beta2_5)],
                # [jnp.array(beta3_0), jnp.array(beta3_1), jnp.array(beta3_2),
                #  jnp.array(beta3_3), jnp.array(beta3_4), jnp.array(beta3_5),
                #  jnp.array(beta3_6), jnp.array(beta3_7), jnp.array(beta3_8),
                #  jnp.array(beta3_9)],
            ])

            ci_[j].append([
                jnp.array(C00),
                jnp.array(C10), jnp.array(C11),
                jnp.array(C20), jnp.array(C21), jnp.array(C22),
                jnp.array(C30), jnp.array(C31), jnp.array(C32), jnp.array(C33),
                jnp.array(C40), jnp.array(C41), jnp.array(C42), jnp.array(C43), jnp.array(C44),
                jnp.array(C50), jnp.array(C51), jnp.array(C52), jnp.array(C53), jnp.array(C54), jnp.array(C55),
            ])

            cicj_[j].append([
                jnp.array(coeff_C0C0), jnp.array(coeff_C0C1), jnp.array(coeff_C0C2), jnp.array(coeff_C0C3), jnp.array(coeff_C0C4), jnp.array(coeff_C0C5),
                jnp.array(coeff_C1C1), jnp.array(coeff_C1C2), jnp.array(coeff_C1C3), jnp.array(coeff_C1C4), jnp.array(coeff_C1C5),
                jnp.array(coeff_C2C2), jnp.array(coeff_C2C3), jnp.array(coeff_C2C4), jnp.array(coeff_C2C5),
                jnp.array(coeff_C3C3), jnp.array(coeff_C3C4), jnp.array(coeff_C3C5),
                jnp.array(coeff_C4C4), jnp.array(coeff_C4C5),
                jnp.array(coeff_C5C5),
            ])

            ci_beta4_[j].append([
                jnp.array(beta_C00),
                jnp.array(beta_C10), jnp.array(beta_C11),
                jnp.array(beta_C20), jnp.array(beta_C21), jnp.array(beta_C22),
                jnp.array(beta_C30), jnp.array(beta_C31), jnp.array(beta_C32), jnp.array(beta_C33),
            ])

            cicj_beta4_[j].append([
                jnp.array(coeff_beta3_C0C0), jnp.array(coeff_beta3_C0C1), jnp.array(coeff_beta3_C0C2), jnp.array(coeff_beta3_C0C3), 
                jnp.array(coeff_beta3_C1C1), jnp.array(coeff_beta3_C1C2), jnp.array(coeff_beta3_C1C3),
                jnp.array(coeff_beta3_C2C2), jnp.array(coeff_beta3_C2C3),
                jnp.array(coeff_beta3_C3C3),
            ])

            dr_[j].append([jnp.array(d0), jnp.array(d1), jnp.array(d2), jnp.array(d3)])

    return cr_, betar_, dr_, ci_, cicj_, ci_beta4_, cicj_beta4_

def compute_smothnesses3_old():
    # BETA_0
    # u_{i-2}^2
    beta0_0 = (
        (
            16.0 * delta_m3p1**2 * delta_m5p1**2 * delta_m1**2 * delta_m2**2 \
            + delta_m3p1**2 * delta_m5p1**2 * delta_m1 * delta_m2**2 * (8.0 * delta_m0 + 16.0 * delta_m2) \
            + delta_m3p1**2 * delta_m5p1**2 * delta_m2**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 *delta_m2 + 4.0 * delta_m2**2) \
            - 16.0 * delta_m3p1 * delta_m5m1 * delta_m5p1**2 * delta_m0 * delta_m1**2 * delta_m2 \
            - delta_m3p1 * delta_m5m1 * delta_m5p1**2 * delta_m0 * delta_m1 * delta_m2 * (12.0 * delta_m0 + 24.0 * delta_m2) \
            - delta_m3p1 * delta_m5m1 * delta_m5p1**2 * delta_m0 * delta_m2 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m2 + 8.0 * delta_m2**2) \
            - 32.0 * delta_m3p1 * delta_m5m1 * delta_m5p1 * delta_m1**2 * delta_m2**2 * (delta_m0 + delta_m1 + delta_m2) \
            - delta_m3p1 * delta_m5m1 * delta_m5p1 * delta_m1 * delta_m2**2 * (84.0 * delta_m0**2 + 16.0 * delta_m0 * delta_m2 + 8.0 * delta_m2**2) \
            + 4.0 * delta_m5m1**2 * delta_m5p1**2 * delta_m0**2 * delta_m1**2 \
            + delta_m5m1**2 * delta_m5p1**2 * delta_m0**2 * delta_m1 * (4.0 * delta_m0 + 8.0 * delta_m2) \
            + delta_m5m1**2 * delta_m5p1**2 * delta_m0**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_m2 + 4.0 * delta_m2**2) \
            + delta_m5m1**2 * delta_m5p1 * delta_m0 * delta_m1**2 * delta_m2 * (20.0 * delta_m0 + 16.0 * delta_m1 + 24.0 * delta_m2) \
            + delta_m5m1**2 * delta_m5p1 * delta_m0 * delta_m1 * delta_m2 * (84.0 * delta_m0**2 + 16.0 * delta_m0 * delta_m2 + 8.0 * delta_m2**2) \
            + delta_m5m1**2 * delta_m1**2 * delta_m2**2 * (48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_m1 + 12.0 * delta_m0 * delta_m2 + 16.0 * delta_m1**2 + 16.0 * delta_m1 * delta_m2 + 4.0 * delta_m2**2)
        )
    ) / (
        delta_m3p1**2 * delta_m5m1**2 * delta_m5p1**2 * delta_m1**2
    )

    # u_{i-2} * u_{i-1}
    beta0_1 = (
        (
            delta_m3p1**2 * delta_m5p1**2 * delta_m1 * delta_m2 * (16.0 * delta_m0 + 32.0 * delta_m1 + 32.0 *delta_m2) \
            + delta_m3p1**2 * delta_m5p1**2 * delta_m2 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m2 + 8.0 * delta_m2**2) \
            - delta_m3p1 * delta_m5m1 * delta_m5p1**2 * delta_m0 * delta_m1 * (12.0 * delta_m0 + 16.0 * delta_m1 + 24.0 * delta_m2) \
            - delta_m3p1 * delta_m5m1 * delta_m5p1**2 * delta_m0 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m2 + 8.0 * delta_m2**2) \
            - delta_m3p1 * delta_m5m1 * delta_m5p1 * delta_m1 * delta_m2 * (168.0 * delta_m0**2 + 64.0 * delta_m0 * delta_m1 + 32.0 * delta_m0 * delta_m2 + 64.0 * delta_m1**2 + 64.0 * delta_m1 * delta_m2 + 16.0 * delta_m2**2) \
            + delta_m5m1**2 * delta_m5p1 * delta_m0 * delta_m1 * (84.0 * delta_m0**2 + 20.0 * delta_m0 * delta_m1 + 16.0 * delta_m0 * delta_m2 + 16.0 * delta_m1**2 + 24.0 * delta_m1 * delta_m2 + 8.0 * delta_m2**2) \
            + delta_m5m1**2 * delta_m1**2 * delta_m2 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_m1 + 24.0 * delta_m0 * delta_m2 + 32.0 * delta_m1**2 + 32.0 * delta_m1 * delta_m2 + 8.0 * delta_m2**2)
            )
    ) / (
        delta_m3p1**2 * delta_m5m1**2 * delta_m5p1**2 * delta_m1
    )
    
    # u_{i-2} * u_{i}
    beta0_2 = (
        delta_m0 * (
            - 32.0 * delta_m3p1 * delta_m5p1 * delta_m1 * delta_m2 * (delta_m0 + delta_m1 + delta_m2) \
            - delta_m3p1 * delta_m5p1 * delta_m2 * (84.0 * delta_m0**2 + 16.0 * delta_m0 * delta_m2 + 8.0 * delta_m2**2) \
            + delta_m5m1 * delta_m5p1 * delta_m0 * delta_m1 * (20.0 * delta_m0 + 16.0 * delta_m1 + 24.0 * delta_m2) \
            + delta_m5m1 * delta_m5p1 * delta_m0 * (84.0 * delta_m0**2 + 16.0 * delta_m0 * delta_m2 + 8.0 * delta_m2**2) \
            + delta_m5m1 * delta_m1 * delta_m2 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_m1 + 24.0 * delta_m0 * delta_m2 + 32.0 * delta_m1**2 + 32.0 * delta_m1 * delta_m2 + 8.0 * delta_m2**2)
        )
    ) / (
        delta_m3p1**2 * delta_m5m1 * delta_m5p1**2 * delta_m1
    )

    # u_{i-1}^2
    beta0_3 = (
        delta_m3p1**2 * delta_m5p1**2 * (40.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 4.0 * delta_m0 * delta_m2 + 16.0 * delta_m1**2 + 16.0 * delta_m1 * delta_m2 + 4.0 * delta_m2**2) \
        - delta_m3p1 * delta_m5m1 * delta_m5p1 * delta_m1 * (84.0 * delta_m0**2 + 32.0 * delta_m0 * delta_m1 + 16.0 * delta_m0 * delta_m2 + 32.0 * delta_m1**2 + 32.0 * delta_m1 * delta_m2 + 8.0 * delta_m2**2) \
        + delta_m5m1**2 * delta_m1**2 * (48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_m1 + 12.0 * delta_m0 * delta_m2 + 16.0 * delta_m1**2 + 16.0 * delta_m1 * delta_m2 + 4.0 * delta_m2**2)
    ) / (
        delta_m3p1**2 * delta_m5m1**2 * delta_m5p1**2
    )
    
    # u_{i} u_{i-1}
    beta0_4 = (
        delta_m0 * (
            - delta_m3p1 * delta_m5p1 * (84.0 * delta_m0**2 + 32.0 * delta_m0 * delta_m1 + 16.0 * delta_m0 * delta_m2 + 32.0 * delta_m1**2 + 32.0 * delta_m1 * delta_m2 + 8.0 * delta_m2**2) \
            + delta_m5m1 * delta_m1 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_m1 + 24.0 * delta_m0 * delta_m2 + 32.0 * delta_m1**2 + 32.0 * delta_m1 * delta_m2 + 8.0 * delta_m2**2)
        )
    ) / (
        delta_m3p1**2 * delta_m5m1 * delta_m5p1**2
    )
    
    # u_{i}**2
    beta0_5 = (
        delta_m0**2 * (
            48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_m1 + 12.0 * delta_m0 * delta_m2 + 16.0 * delta_m1**2 + 16.0 * delta_m1 * delta_m2 + 4.0 * delta_m2**2
        )
    ) / (
        delta_m3p1**2 * delta_m5p1**2
    )
    
    # BETA_1
    # u_{i-1}**2
    beta1_0 = (
        4.0 * delta_m1p3**2 * delta_m3p3**2 * delta_p1**2 * delta_m1**2 \
        - delta_m1p3**2 * delta_m3p3**2 * delta_p1 * delta_m1**2 * (4.0 * delta_m0 + 8.0 * delta_m1) \
        + delta_m1p3**2 * delta_m3p3**2 * delta_m1**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_m1 + 4.0 * delta_m1**2) \
        + delta_m1p3 * delta_m3p1 * delta_m3p3**2 * delta_p1**2 * delta_m1 * (-8.0 * delta_p1 + 16.0 * delta_m1) \
        - delta_m1p3 * delta_m3p1 * delta_m3p3**2 * delta_p1 * delta_m1 * (76.0 * delta_m0**2 + 8.0 * delta_m1**2) \
        + delta_m1p3 * delta_m3p1 * delta_m3p3 * delta_m0 * delta_p1 * delta_m1**2 * (4.0 * delta_m0 + 8.0 * delta_m1) \
        - delta_m1p3 * delta_m3p1 * delta_m3p3 * delta_m0 * delta_m1**2 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2) \
        + delta_m3p1**2 * delta_m3p3**2 * delta_p1**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_p1 - 4.0 * delta_m0 * delta_m1 + 4.0 * delta_p1**2 - 8.0 * delta_p1 * delta_m1 + 4.0 * delta_m1**2) \
        - delta_m3p1**2 * delta_m3p3 * delta_m0 * delta_p1**2 * delta_m1 * (4.0 * delta_m0 + 8.0 * delta_m1) \
        + delta_m3p1**2 * delta_m3p3 * delta_m0 * delta_p1 * delta_m1 * (76.0 * delta_m0**2 + 8.0 * delta_m1**2) \
        + delta_m3p1**2 * delta_m0**2 * delta_m1**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_m1 + 4.0 * delta_m1**2)
    ) / (
        delta_m1p3**2 * delta_m3p1**2 * delta_m3p3**2 * delta_p1**2
    )

    # u_{i} u_{i-1}
    beta1_1 = (
        delta_m0 * (
            8.0 * delta_m1p3**2 * delta_m3p3**2 * delta_p1**2 * delta_m1 \
            - delta_m1p3**2 * delta_m3p3**2 * delta_p1 * delta_m1 * (8.0 * delta_m0 + 16.0 * delta_m1) \
            + delta_m1p3**2 * delta_m3p3**2 * delta_m1 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2) \
            + delta_m1p3 * delta_m3p1 * delta_m3p3**2 * delta_p1**2 * (-8.0 * delta_p1 + 16.0 * delta_m1) \
            - delta_m1p3 * delta_m3p1 * delta_m3p3**2 * delta_p1 * (76.0 * delta_m0**2 + 8.0 * delta_m1**2) \
            + delta_m1p3 * delta_m3p1 * delta_m3p3 * delta_m0 * delta_p1 * delta_m1 * (8.0 * delta_m0 + 16.0 * delta_m1) \
            - delta_m1p3 * delta_m3p1 * delta_m3p3 * delta_m0 * delta_m1 * (160.0 * delta_m0**2 + 16.0 * delta_m0 * delta_m1 + 16.0 * delta_m1**2) \
            - delta_m3p1**2 * delta_m3p3 * delta_m0 * delta_p1**2 * (4.0 * delta_m0 + 8.0 * delta_m1) \
            + delta_m3p1**2 * delta_m3p3 * delta_m0 * delta_p1 * (76.0 * delta_m0**2 + 8.0 * delta_m1**2) \
            + delta_m3p1**2 * delta_m0**2 * delta_m1 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2)
            )
        ) / (
            delta_m1p3**2 * delta_m3p1**2 * delta_m3p3**2 * delta_p1**2
        )
    
    # u_{i+1} u_{i-1}
    beta1_2 = (
            delta_m0 * (
                delta_m1p3 * delta_m3p3 * delta_p1 * delta_m1 * (4.0 * delta_m0 + 8.0 * delta_m1) \
                - delta_m1p3 * delta_m3p3 * delta_m1 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2) \
                + delta_m3p1 * delta_m3p3 * delta_p1 * (76.0 * delta_m0**2 - 4.0 * delta_m0 * delta_p1 - 8.0 * delta_p1 * delta_m1 + 8.0 * delta_m1**2) \
                + delta_m3p1 * delta_m0 * delta_m1 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2)
            )
        ) / (
            delta_m1p3**2 * delta_m3p1 * delta_m3p3**2 * delta_p1
        )
    
    # u_{i}**2
    beta1_3 = (
            delta_m0**2 * (
                4.0 * delta_m1p3**2 * delta_m3p3**2 * delta_p1**2 \
                - delta_m1p3**2 * delta_m3p3**2 * delta_p1 * (4.0 * delta_m0 + 8.0 * delta_m1) \
                + delta_m1p3**2 * delta_m3p3**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_m1 + 4.0 * delta_m1**2) \
                + delta_m1p3 * delta_m3p1 * delta_m3p3 * delta_m0 * delta_p1 * (4.0 * delta_m0 + 8.0 * delta_m1) \
                - delta_m1p3 * delta_m3p1 * delta_m3p3 * delta_m0 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2) \
                + delta_m3p1**2 * delta_m0**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_m1 + 4.0 * delta_m1**2)
            )
        ) / (
            delta_m1p3**2 * delta_m3p1**2 * delta_m3p3**2 * delta_p1**2
        )
    
    # u_{i} u_{i+1}
    beta1_4 = (
            delta_m0**2 * (
                delta_m1p3 * delta_m3p3 * delta_p1 * (4.0 * delta_m0 + 8.0 * delta_m1) \
                - delta_m1p3 * delta_m3p3 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2) \
                + delta_m3p1 * delta_m0 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_m1 + 8.0 * delta_m1**2)
            )
        ) / (
            delta_m1p3**2 * delta_m3p1 * delta_m3p3**2 * delta_p1 
        )
    
    # u_{i+1}**2
    beta1_5 = (
            delta_m0**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_m1 + 4.0 * delta_m1**2)
        ) / (
            delta_m1p3**2 * delta_m3p3**2
        )
    
    # BETA_2
    # u_{i}**2
    beta2_0 = (
        delta_m0**2 * (
            16.0 * delta_m1p3**2 * delta_m1p5**2 * delta_p1**2 * delta_p2**2 \
            + delta_m1p3**2 * delta_m1p5**2 * delta_p1 * delta_p2**2 * (8.0 * delta_m0 + 16.0 * delta_p2) \
            + delta_m1p3**2 * delta_m1p5**2 * delta_p2**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_p2 + 4.0 * delta_p2**2) \
            + 80.0 * delta_m1p3**2 * delta_m1p5 * delta_m0**3 * delta_p1 * delta_p2 \
            + 4.0 * delta_m1p3**2 * delta_m1p5 * delta_m0**2 * delta_p1 * delta_p2**2 \
            + 8.0 * delta_m1p3**2 * delta_m1p5 * delta_m0 * delta_p1**2 * delta_p2**2 \
            + delta_m1p3**2 * delta_m1p5 * delta_m0 * delta_p1**2 * delta_p2 * (12.0 * delta_m0 + 16.0 * delta_p1) \
            + delta_m1p3**2 * delta_m0**2 * delta_p1**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_p1 + 4.0 * delta_p1**2) \
            - 80.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_m0**3 * delta_p2 \
            - 12.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_m0**2 * delta_p1 * delta_p2 \
            - 16.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_m0 * delta_p1**2 * delta_p2 \
            - 24.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_m0 * delta_p1 * delta_p2**2 \
            - 8.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_m0 * delta_p2**2 * (delta_m0 + delta_p2) \
            - 80.0 * delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_m0**4 * delta_p1 \
            - 4.0 * delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_m0**3 * delta_p1 * delta_p2 \
            - 8.0 * delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_m0**2 * delta_p1**2 * delta_p2 \
            - 8.0 * delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_m0**2 * delta_p1**2 * (delta_m0 + delta_p1) \
            + 40.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**4 \
            + 4.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**3 * delta_p1 \
            + 4.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**3 * delta_p2 \
            + 4.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**2 * delta_p1**2 \
            + 8.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**2 * delta_p1 * delta_p2 \
            + 4.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**2 * delta_p2**2
            )
        ) / (
            delta_m1p3**2 * delta_m1p5**2 * delta_p1p5**2 * delta_p1**2 * delta_p2**2
        )
    
    # u_{i} u_{i+1}
    beta2_1 = (
        delta_m0**2 * (
            delta_m1p3**2 * delta_m1p5 * delta_p1 * delta_p2**2 * (4.0 * delta_m0 + 8.0 * delta_p1) \
            + delta_m1p3**2 * delta_m1p5 * delta_p1 * delta_p2 * (80.0 * delta_m0**2 + 12.0 * delta_m0 * delta_p1 + 16.0 * delta_p1**2) \
            + delta_m1p3**2 * delta_m0 * delta_p1**2 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_p1 + 8.0 * delta_p1**2) \
            - 80.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_m0**2 * delta_p2 \
            - 24.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_p1 * delta_p2**2 \
            - delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_p1 * delta_p2 * (12.0 * delta_m0 + 16.0 * delta_p1) \
            - 8.0 * delta_m1p3 * delta_m1p5**2 * delta_p1p5 * delta_p2**2 * (delta_m0 + delta_p2) \
            - delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_m0 * delta_p1 * delta_p2 * (8.0 * delta_m0 + 16.0 * delta_p1) \
            - delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_m0 * delta_p1 * (160.0 * delta_m0**2 + 16.0 * delta_m0 * delta_p1 + 16.0 * delta_p1**2) \
            + 80.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**3 \
            + 8.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0**2 * delta_p2 \
            + 16.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0 * delta_p1 * delta_p2 \
            + 8.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0 * delta_p1 * (delta_m0 + delta_p1) \
            + 8.0 * delta_m1p5**2 * delta_p1p5**2 * delta_m0 * delta_p2**2
            )
        ) / (
            delta_m1p3**2 * delta_m1p5**2 * delta_p1p5**2 * delta_p1 * delta_p2**2
        )
    
    # u_{i} u_{i+2}
    beta2_2 = (
        delta_m0**2 * (
            delta_m1p3 * delta_m1p5 * delta_m0 * delta_p2 * (80.0 * delta_m0 + 4.0 * delta_p2) \
            + delta_m1p3 * delta_m1p5 * delta_p1 * delta_p2 * (12.0 * delta_m0 + 16.0 * delta_p1 + 8.0 * delta_p2) \
            + delta_m1p3 * delta_m0 * delta_p1 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_p1 + 8.0 * delta_p1**2) \
            - 80.0 * delta_m1p5 * delta_p1p5 * delta_m0**3 \
            - 4.0 * delta_m1p5 * delta_p1p5 * delta_m0**2 * delta_p2 \
            - 8.0 * delta_m1p5 * delta_p1p5 * delta_m0 * delta_p1 * delta_p2 \
            - 8.0 * delta_m1p5 * delta_p1p5 * delta_m0 * delta_p1 * (delta_m0 + delta_p1)
            )
        ) / (
            delta_m1p3 * delta_m1p5**2 * delta_p1p5**2 * delta_p1 * delta_p2
        )

    # u_{i+1}**2
    beta2_3 = (
        delta_m0**2 * (
            delta_m1p3**2 * delta_p1**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_p1 + 4.0 * delta_p1**2) \
            - delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_p1 * delta_p2 * (4.0 * delta_m0 + 8.0 * delta_p1) \
            - delta_m1p3 * delta_m1p5 * delta_p1p5 * delta_p1 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_p1 + 8.0 * delta_p1**2) \
            + 4.0 * delta_m1p5**2 * delta_p1p5**2 * delta_p2**2 \
            + delta_m1p5**2 * delta_p1p5**2 * delta_p2 * (4.0 * delta_m0 + 8.0 * delta_p1) \
            + delta_m1p5**2 * delta_p1p5**2 * (40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_p1 + 4.0 * delta_p1**2)
            )
        ) / (
            delta_m1p3**2 * delta_m1p5**2 * delta_p1p5**2 * delta_p2**2
        )
    
    # u_{i+1} u_{i+2}
    beta2_4 = (
        delta_m0**2 * (
            delta_m1p3 * delta_p1 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_p1 + 8.0 * delta_p1**2) \
            - delta_m1p5 * delta_p1p5 * delta_p2 * (4.0 * delta_m0 + 8.0 * delta_p1) \
            - delta_m1p5 * delta_p1p5 * (80.0 * delta_m0**2 + 8.0 * delta_m0 * delta_p1 + 8.0 * delta_p1**2)
            )
        ) / (
            delta_m1p3 * delta_m1p5**2 * delta_p1p5**2 * delta_p2
        )

    # u_{i+2}**2
    beta2_5 = (
        delta_m0**2 * (
            40.0 * delta_m0**2 + 4.0 * delta_m0 * delta_p1 + 4.0 * delta_p1**2
            )
        ) / (
            delta_m1p5**2 * delta_p1p5**2
        )

def compute_smothnesses3(delta_m2, delta_m1, delta_m0, delta_p1, delta_p2, delta_p3):

    delta_x0 = delta_m2 
    delta_x1 = delta_m1 
    delta_x2 = delta_m0 
    delta_x3 = delta_p1 
    delta_x4 = delta_p2
    delta_x5 = delta_p3

    (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
    (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
    (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
    = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1, delta_p2, delta_p3)

    # BETA_03
    one_beta_0 = 1.0 / ((delta_x2 + delta_x1)**2 * \
        (delta_x1 + delta_x0)**2 * \
        (delta_x2 + delta_x1 + delta_x0)**2)
    # u_{i-2}^2
    beta0_0 = one_beta_0 * delta_x2**2 * (
        40.0 * delta_x2**4 + \
        84.0 * delta_x2**3 * delta_x1 + \
        52.0 * delta_x2**2 * delta_x1**2 + \
        12.0 * delta_x2 * delta_x1**3 + \
        4.0 * delta_x1**4
    )
    # u_{i-2} * u_{i-1}
    beta0_1 = one_beta_0 * delta_x2**2 * (
        -80.0 * delta_x2**4 - \
        252.0 * delta_x2**3 * delta_x1 - \
        84.0  * delta_x2**3 * delta_x0 - \
        208.0 * delta_x2**2 * delta_x1**2 - \
        108.0 * delta_x2**2 * delta_x1 * delta_x0 - \
        4.0   * delta_x2**2 * delta_x0**2 - \
        60.0  * delta_x2 * delta_x1**3 - \
        48.0  * delta_x2 * delta_x1**2 * delta_x0 - \
        12.0  * delta_x2 * delta_x1 * delta_x0**2 - \
        24.0  * delta_x1**4 - \
        24.0  * delta_x1**3 * delta_x0 - \
        8.0   * delta_x1**2 * delta_x0**2
    )
    # u_{i-2} * u_{i}
    beta0_2 = one_beta_0 * delta_x2**2 * (
        84.0  * delta_x2**3 * delta_x1 + \
        84.0  * delta_x2**3 * delta_x0 + \
        104.0 * delta_x2**2 * delta_x1**2 + \
        108.0 * delta_x2**2 * delta_x1 * delta_x0 + \
        4.0   * delta_x2**2 * delta_x0**2 + \
        36.0  * delta_x2 * delta_x1**3 + \
        48.0  * delta_x2 * delta_x1**2 * delta_x0 + \
        12.0  * delta_x2 * delta_x1 * delta_x0**2 + \
        16.0  * delta_x1**4 + \
        24.0  * delta_x1**3 * delta_x0 + \
        8.0   * delta_x1**2 * delta_x0**2
    )
    # u_{i-1}^2
    beta0_3 = one_beta_0 * delta_x2**2 * (
        40.0  * delta_x2**4 + \
        168.0 * delta_x2**3 * delta_x1 + \
        84.0  * delta_x2**3 * delta_x0 + \
        204.0 * delta_x2**2 * delta_x1**2 + \
        204.0 * delta_x2**2 * delta_x1 * delta_x0 + \
        52.0  * delta_x2**2 * delta_x0**2 + \
        72.0  * delta_x2 * delta_x1**3 + \
        108.0 * delta_x2 * delta_x1**2 * delta_x0 + \
        60.0  * delta_x2 * delta_x1 * delta_x0**2 + \
        12.0  * delta_x2 * delta_x0**3 + \
        36.0  * delta_x1**4 + \
        72.0  * delta_x1**3 * delta_x0 + \
        60.0  * delta_x1**2 * delta_x0**2 + \
        24.0  * delta_x1 * delta_x0**3 + \
        4.0   * delta_x0**4
    )
    # u_{i} u_{i-1}
    beta0_4 = one_beta_0 * delta_x2**2 * (
        -84.0 * delta_x2**3 * delta_x1 - \
        84.0  * delta_x2**3 * delta_x0 - \
        200.0 * delta_x2**2 * delta_x1**2 - \
        300.0 * delta_x2**2 * delta_x1 * delta_x0 - \
        100.0 * delta_x2**2 * delta_x0**2 - \
        84.0  * delta_x2 * delta_x1**3 - \
        168.0 * delta_x2 * delta_x1**2 * delta_x0 - \
        108.0 * delta_x2 * delta_x1 * delta_x0**2 - \
        24.0  * delta_x2 * delta_x0**3 - \
        48.0  * delta_x1**4 - \
        120.0 * delta_x1**3 * delta_x0 - \
        112.0 * delta_x1**2 * delta_x0**2 - \
        48.0  * delta_x1 * delta_x0**3 - \
        8.0   * delta_x0**4
    )
    # u_{i}**2
    beta0_5 = one_beta_0 * delta_x2**2 * (
        48.0 * delta_x2**2 * delta_x1**2 + \
        96.0 * delta_x2**2 * delta_x1 * delta_x0 + \
        48.0 * delta_x2**2 * delta_x0**2 + \
        24.0 * delta_x2 * delta_x1**3 + \
        60.0 * delta_x2 * delta_x1**2 * delta_x0 + \
        48.0 * delta_x2 * delta_x1 * delta_x0**2 + \
        12.0 * delta_x2 * delta_x0**3 + \
        16.0 * delta_x1**4 + \
        48.0 * delta_x1**3 * delta_x0 + \
        52.0 * delta_x1**2 * delta_x0**2 + \
        24.0 * delta_x1 * delta_x0**3 + \
        4.0  * delta_x0**4
    )

    # BETA_13
    one_beta_1 = 1.0 / ((delta_x2 + delta_x3)**2 * \
        (delta_x2 + delta_x1)**2 * \
        (delta_x2 + delta_x3 + delta_x1)**2)
    # u_{i-1}**2
    beta1_0 = one_beta_1 * delta_x2**2 * (
        40.0 * delta_x2**4 + \
        84.0 * delta_x2**3 * delta_x3 + \
        52.0 * delta_x2**2 * delta_x3**2 + \
        12.0 * delta_x2 * delta_x3**3 + \
        4.0  * delta_x3**4
    )
    # u_{i} u_{i-1}
    beta1_1 = one_beta_1 * delta_x2**2 * (
        -156.0 * delta_x2**4 - \
        240.0  * delta_x2**3 * delta_x3 - \
        72.0   * delta_x2**3 * delta_x1 - \
        100.0  * delta_x2**2 * delta_x3**2 - \
        60.0   * delta_x2**2 * delta_x3 * delta_x1 + \
        4.0    * delta_x2**2 * delta_x1**2 - \
        24.0   * delta_x2 * delta_x3**3 + \
        12.0   * delta_x2 * delta_x3**2 * delta_x1 + \
        12.0   * delta_x2 * delta_x3 * delta_x1**2 - \
        8.0    * delta_x3**4 + \
        8.0    * delta_x3**2 * delta_x1**2
    )
    # u_{i+1} u_{i-1}
    beta1_2 = one_beta_1 * delta_x2**2 * (
        76.0 * delta_x2**4 + \
        72.0 * delta_x2**3 * delta_x3 + \
        72.0 * delta_x2**3 * delta_x1 - \
        4.0  * delta_x2**2 * delta_x3**2 + \
        60.0 * delta_x2**2 * delta_x3 * delta_x1 - \
        4.0  * delta_x2**2 * delta_x1**2 - \
        12.0 * delta_x2 * delta_x3**2 * delta_x1 - \
        12.0 * delta_x2 * delta_x3 * delta_x1**2 - \
        8.0  * delta_x3**2 * delta_x1**2
    )
    # u_{i}**2
    beta1_3 = one_beta_1 * delta_x2**2 * (
        156.0 * delta_x2**4 + \
        156.0 * delta_x2**3 * delta_x3 + \
        156.0 * delta_x2**3 * delta_x1 + \
        48.0  * delta_x2**2 * delta_x3**2 + \
        60.0  * delta_x2**2 * delta_x3 * delta_x1 + \
        48.0  * delta_x2**2 * delta_x1**2 + \
        12.0  * delta_x2 * delta_x3**3 - \
        12.0  * delta_x2 * delta_x3**2 * delta_x1 - \
        12.0  * delta_x2 * delta_x3 * delta_x1**2 + \
        12.0  * delta_x2 * delta_x1**3 + \
        4.0   * delta_x3**4 - \
        8.0   * delta_x3**2 * delta_x1**2 + \
        4.0   * delta_x1**4
    )
    # u_{i} u_{i+1}
    beta1_4 = one_beta_1 * delta_x2**2 * (
        -156.0 * delta_x2**4 - \
        72.0   * delta_x2**3 * delta_x3 - \
        240.0  * delta_x2**3 * delta_x1 + \
        4.0    * delta_x2**2 * delta_x3**2 - \
        60.0   * delta_x2**2 * delta_x3 * delta_x1 - \
        100.0  * delta_x2**2 * delta_x1**2 + \
        12.0   * delta_x2 * delta_x3**2 * delta_x1 + \
        12.0   * delta_x2 * delta_x3 * delta_x1**2 - \
        24.0   * delta_x2 * delta_x1**3 + \
        8.0    * delta_x3**2 * delta_x1**2 - \
        8.0    * delta_x1**4
    )
    # u_{i+1}**2
    beta1_5 = one_beta_1 * delta_x2**2 * (
        40.0 * delta_x2**4 + \
        84.0 * delta_x2**3 * delta_x1 + \
        52.0 * delta_x2**2 * delta_x1**2 + \
        12.0 * delta_x2 * delta_x1**3 + \
        4.0  * delta_x1**4
    )

    # BETA_23
    one_beta_2 = 1.0 / ((delta_x2 + delta_x3)**2 * \
        (delta_x3 + delta_x4)**2 * \
        (delta_x2 + delta_x3 + delta_x4)**2)
    # u_{i}**2
    beta2_0 = one_beta_2 * delta_x2**2 * (
        48.0 * delta_x2**2 * delta_x3**2 + \
        96.0 * delta_x2**2 * delta_x3 * delta_x4 + \
        48.0 * delta_x2**2 * delta_x4**2 + \
        24.0 * delta_x2 * delta_x3**3 + \
        60.0 * delta_x2 * delta_x3**2 * delta_x4 + \
        48.0 * delta_x2 * delta_x3 * delta_x4**2 + \
        12.0 * delta_x2 * delta_x4**3 + \
        16.0 * delta_x3**4 + \
        48.0 * delta_x3**3 * delta_x4 + \
        52.0 * delta_x3**2 * delta_x4**2 + \
        24.0 * delta_x3 * delta_x4**3 + \
        4.0  * delta_x4**4
    )
    # u_{i} u_{i+1}
    beta2_1 = one_beta_2 * delta_x2**2 * (
        -84.0 * delta_x2**3 * delta_x3 - \
        84.0  * delta_x2**3 * delta_x4 - \
        200.0 * delta_x2**2 * delta_x3**2 - \
        300.0 * delta_x2**2 * delta_x3 * delta_x4 - \
        100.0 * delta_x2**2 * delta_x4**2 - \
        84.0  * delta_x2 * delta_x3**3 - \
        168.0 * delta_x2 * delta_x3**2 * delta_x4 - \
        108.0 * delta_x2 * delta_x3 * delta_x4**2 - \
        24.0  * delta_x2 * delta_x4**3 - \
        48.0  * delta_x3**4 - \
        120.0 * delta_x3**3 * delta_x4 - \
        112.0 * delta_x3**2 * delta_x4**2 - \
        48.0  * delta_x3 * delta_x4**3 - \
        8.0   * delta_x4**4
    )
    # u_{i} u_{i+2}
    beta2_2 = one_beta_2 * delta_x2**2 * (
        84.0  * delta_x2**3 * delta_x3 + \
        84.0  * delta_x2**3 * delta_x4 + \
        104.0 * delta_x2**2 * delta_x3**2 + \
        108.0 * delta_x2**2 * delta_x3 * delta_x4 + \
        4.0   * delta_x2**2 * delta_x4**2 + \
        36.0  * delta_x2 * delta_x3**3 + \
        48.0  * delta_x2 * delta_x3**2 * delta_x4 + \
        12.0  * delta_x2 * delta_x3 * delta_x4**2 + \
        16.0  * delta_x3**4 + \
        24.0  * delta_x3**3 * delta_x4 + \
        8.0   * delta_x3**2 * delta_x4**2
    )
    # u_{i+1}**2
    beta2_3 = one_beta_2 * delta_x2**2 * (
        40.0  * delta_x2**4 + \
        168.0 * delta_x2**3 * delta_x3 + \
        84.0  * delta_x2**3 * delta_x4 + \
        204.0 * delta_x2**2 * delta_x3**2 + \
        204.0 * delta_x2**2 * delta_x3 * delta_x4 + \
        52.0  * delta_x2**2 * delta_x4**2 + \
        72.0  * delta_x2 * delta_x3**3 + \
        108.0 * delta_x2 * delta_x3**2 * delta_x4 + \
        60.0  * delta_x2 * delta_x3 * delta_x4**2 + \
        12.0  * delta_x2 * delta_x4**3 + \
        36.0  * delta_x3**4 + \
        72.0  * delta_x3**3 * delta_x4 + \
        60.0  * delta_x3**2 * delta_x4**2 + \
        24.0  * delta_x3 * delta_x4**3 + \
        4.0   * delta_x4**4
    )
    # u_{i+1} u_{i+2}
    beta2_4 = one_beta_2 * delta_x2**2 * (
        -80.0 * delta_x2**4 - \
        252.0 * delta_x2**3 * delta_x3 - \
        84.0  * delta_x2**3 * delta_x4 - \
        208.0 * delta_x2**2 * delta_x3**2 - \
        108.0 * delta_x2**2 * delta_x3 * delta_x4 - \
        4.0   * delta_x2**2 * delta_x4**2 - \
        60.0  * delta_x2 * delta_x3**3 - \
        48.0  * delta_x2 * delta_x3**2 * delta_x4 - \
        12.0  * delta_x2 * delta_x3 * delta_x4**2 - \
        24.0  * delta_x3**4 - \
        24.0  * delta_x3**3 * delta_x4 - \
        8.0   * delta_x3**2 * delta_x4**2
    )
    # u_{i+2}**2
    beta2_5 = one_beta_2 * delta_x2**2 * (
        40.0 * delta_x2**4 + \
        84.0 * delta_x2**3 * delta_x3 + \
        52.0 * delta_x2**2 * delta_x3**2 + \
        12.0 * delta_x2 * delta_x3**3 + \
        4.0  * delta_x3**4
    )

    # BETA_33
    # u_{i+1}**2
    beta3_0 = (
        delta_m0**2 * (
            16.0 * delta_p1p5**2 * delta_p1p7**2 * delta_p2**2 * delta_p3**2 \
            + delta_p1p5**2 * delta_p1p7**2 * delta_p2 * delta_p3**2 * (24.0 * delta_m0 + 32.0 * delta_p1 + 16.0 * delta_p3) \
            + delta_p1p5**2 * delta_p1p7**2 * delta_p3**2 * (48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_p1 + 12.0 * delta_m0 * delta_p3 + 16.0 * delta_p1**2 + 16.0 * delta_p1 * delta_p3 + 4.0 * delta_p3**2) \
            + 8.0 * delta_p1p5**2 * delta_p1p7 * delta_p1 * delta_p2**2 * delta_p3**2 \
            + delta_p1p5**2 * delta_p1p7 * delta_p1 * delta_p2**2 * delta_p3 * (36.0 * delta_m0 + 48.0 * delta_p1 + 16.0 * delta_p2) \
            + delta_p1p5**2 * delta_p1p7 * delta_p1 * delta_p2 * delta_p3**2 * (12.0 * delta_m0 + 16.0 * delta_p1) \
            + delta_p1p5**2 * delta_p1p7 * delta_p1 * delta_p2 * delta_p3 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 32.0 * delta_p1**2) \
            + delta_p1p5**2 * delta_p1**2 * delta_p2**2 * (48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_p1 + 12.0 * delta_m0 * delta_p2 + 16.0 * delta_p1**2 + 16.0 * delta_p1 * delta_p2 + 4.0 * delta_p2**2) \
            - 16.0 * delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p1 * delta_p2**2 * delta_p3 \
            - 24.0 * delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p1 * delta_p2 * delta_p3**2 \
            - delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p1 * delta_p2 * delta_p3 * (36.0 * delta_m0 + 48.0 * delta_p1) \
            - delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p1 * delta_p3**2 * (24.0 * delta_m0 + 32.0 * delta_p1 + 8.0 * delta_p3) \
            - delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p1 * delta_p3 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 32.0 * delta_p1**2) \
            - 8.0 * delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p1**2 * delta_p2**2 * delta_p3 \
            - delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p1**2 * delta_p2**2 * (24.0 * delta_m0 + 32.0 * delta_p1 + 8.0 * delta_p2) \
            - delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p1**2 * delta_p2 * delta_p3 * (12.0 * delta_m0 + 16.0 * delta_p1) \
            - delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p1**2 * delta_p2 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 32.0 * delta_p1**2) \
            + 4.0 * delta_p1p7**2 * delta_p3p7**2 * delta_p1**2 * delta_p2**2 \
            + 8.0 * delta_p1p7**2 * delta_p3p7**2 * delta_p1**2 * delta_p2 * delta_p3 \
            + delta_p1p7**2 * delta_p3p7**2 * delta_p1**2 * delta_p2 * (12.0 * delta_m0 + 16.0 * delta_p1) \
            + 4.0 * delta_p1p7**2 * delta_p3p7**2 * delta_p1**2 * delta_p3**2 \
            + delta_p1p7**2 * delta_p3p7**2 * delta_p1**2 * delta_p3 * (12.0 * delta_m0 + 16.0 * delta_p1) \
            + delta_p1p7**2 * delta_p3p7**2 * delta_p1**2 * (48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_p1 + 16.0 * delta_p1**2)
            )
        ) / (
            delta_p1p5**2 * delta_p1p7**2 * delta_p3p7**2 * delta_p2**2 * delta_p3**2 
        )
    
    # u_{i+1} u_{i+2}
    beta3_1 = (
        delta_m0**2 * (
            delta_p1p5**2 * delta_p1p7 * delta_p2 * delta_p3**2 * (12.0 * delta_m0 + 16.0 * delta_p1 + 8.0 * delta_p2) \
            + delta_p1p5**2 * delta_p1p7 * delta_p2 * delta_p3 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 36.0 * delta_m0 * delta_p2 + 32.0 * delta_p1**2 + 48.0 * delta_p1 * delta_p2 + 16.0 * delta_p2**2) \
            + delta_p1p5**2 * delta_p1 * delta_p2**2 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 24.0 * delta_m0 * delta_p2 + 32.0 * delta_p1**2 + 32.0 * delta_p1 * delta_p2 + 8.0 * delta_p2**2) \
            - 24.0 * delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p2 * delta_p3**2 \
            - delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p2 * delta_p3 * (36.0 * delta_m0 + 48.0 * delta_p1 + 16.0 * delta_p2) \
            - delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p3**2 * (24.0 * delta_m0 + 32.0 * delta_p1 + 8.0 * delta_p3) \
            - delta_p1p5 * delta_p1p7**2 * delta_p3p7 * delta_p3 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 32.0 * delta_p1**2) \
            - delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p1 * delta_p2 * delta_p3 * (24.0 * delta_m0 + 32.0 * delta_p1 + 16.0 * delta_p2) \
            - delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p1 * delta_p2 * (192.0 * delta_m0**2 + 96.0 * delta_m0 * delta_p1 + 48.0 * delta_m0 * delta_p2 + 64.0 * delta_p1**2 + 64.0 * delta_p1 * delta_p2 + 16.0 * delta_p2**2) \
            + 16.0 * delta_p1p7**2 * delta_p3p7**2 * delta_p1 * delta_p2 * delta_p3 \
            + delta_p1p7**2 * delta_p3p7**2 * delta_p1 * delta_p2 * (24.0 * delta_m0 + 32.0 * delta_p1 + 8.0 * delta_p2) \
            + 8.0 * delta_p1p7**2 * delta_p3p7**2 * delta_p1 * delta_p3**2 \
            + delta_p1p7**2 * delta_p3p7**2 * delta_p1 * delta_p3 * (24.0 * delta_m0 + 32.0 * delta_p1) \
            + delta_p1p7**2 * delta_p3p7**2 * delta_p1 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 32.0 * delta_p1**2)
            )
        ) / (
            delta_p1p5**2 * delta_p1p7**2 * delta_p3p7**2 * delta_p2 * delta_p3**2
        )
    
    # u_{i+1} u_{i+3}
    beta3_2 = (
        delta_m0**2 * (
            delta_p1p5 * delta_p1p7 * delta_p2 * delta_p3 * (36.0 * delta_m0 + 48.0 * delta_p1 + 16.0 * delta_p2 + 8.0 * delta_p3) \
            + delta_p1p5 * delta_p1p7 * delta_p3 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 12.0 * delta_m0 * delta_p3 + 32.0 * delta_p1**2 + 16.0 * delta_p1 * delta_p3) \
            + delta_p1p5 * delta_p1 * delta_p2 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 24.0 * delta_m0 * delta_p2 + 32.0 * delta_p1**2 + 32.0 * delta_p1 * delta_p2 + 8.0 * delta_p2**2) \
            - 8.0 * delta_p1p7 * delta_p3p7 * delta_p1 * delta_p2 * delta_p3 \
            - delta_p1p7 * delta_p3p7 * delta_p1 * delta_p2 * (24.0 * delta_m0 + 32.0 * delta_p1 + 8.0 * delta_p2) \
            - delta_p1p7 * delta_p3p7 * delta_p1 * delta_p3 * (12.0 * delta_m0 + 16.0 * delta_p1) \
            - delta_p1p7 * delta_p3p7 * delta_p1 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 32.0 * delta_p1**2)
            )
        ) / (
            delta_p1p5 * delta_p1p7**2 * delta_p3p7**2 * delta_p2 * delta_p3
        )

    # u_{i+2}**2
    beta3_3 = (
        delta_m0**2 * (
            delta_p1p5**2 * delta_p2**2 * (48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_p1 + 12.0 * delta_m0 * delta_p2 + 16.0 * delta_p1**2 + 16.0 * delta_p1 * delta_p2 + 4.0 * delta_p2**2) \
            - delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p2 * delta_p3 * (12.0 * delta_m0 + 16.0 * delta_p1 + 8.0 * delta_p2) \
            - delta_p1p5 * delta_p1p7 * delta_p3p7 * delta_p2 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 24.0 * delta_m0 * delta_p2 + 32.0 * delta_p1**2 + 32.0 * delta_p1 * delta_p2 + 8.0 * delta_p2**2) \
            + 4.0 * delta_p1p7**2 * delta_p3p7**2 * delta_p3**2 \
            + delta_p1p7**2 * delta_p3p7**2 * delta_p3 * (12.0 * delta_m0 + 16.0 * delta_p1 + 8.0 * delta_p2) \
            + delta_p1p7**2 * delta_p3p7**2 * (48.0 * delta_m0**2 + 24.0 * delta_m0 * delta_p1 + 12.0 * delta_m0 * delta_p2 + 16.0 * delta_p1**2 + 16.0 * delta_p1 * delta_p2 + 4.0 * delta_p2**2 )
            )
        ) / (
            delta_p1p5**2 * delta_p1p7**2 * delta_p3p7**2 * delta_p3**2
        )
    
    # u_{i+2} u_{i+3}
    beta3_4 = (
        delta_m0**2 * (
            delta_p1p5 * delta_p2 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 24.0 * delta_m0 * delta_p2 + 32.0 * delta_p1**2 + 32.0 * delta_p1 * delta_p2 + 8.0 * delta_p2**2) \
            - delta_p1p7 * delta_p3p7 * delta_p3 * (12.0 * delta_m0 + 16.0 * delta_p1 + 8.0 * delta_p2) \
            - delta_p1p7 * delta_p3p7 * (96.0 * delta_m0**2 + 48.0 * delta_m0 * delta_p1 + 24.0 * delta_m0 * delta_p2 + 32.0 * delta_p1**2 + 32.0 * delta_p1 * delta_p2 + 8.0 * delta_p2**2)
            )
        ) / (
            delta_p1p5 * delta_p1p7**2 * delta_p3p7**2 * delta_p3
        )
    
    # u_{i+3}**2
    beta3_5 = (
        delta_m0**2 * (
            48.0 * delta_m0**2 \
            + 24.0 * delta_m0 * delta_p1 \
            + 12.0 * delta_m0 * delta_p2 \
            + 16.0 * delta_p1**2 \
            + 16.0 * delta_p1 * delta_p2 \
            + 4.0 * delta_p2**2
            )
        ) / (
            delta_p1p7**2 * delta_p3p7**2
        )

    return (beta0_0, beta0_1, beta0_2, beta0_3, beta0_4, beta0_5), \
        (beta1_0, beta1_1, beta1_2, beta1_3, beta1_4, beta1_5), \
        (beta2_0, beta2_1, beta2_2, beta2_3, beta2_4, beta2_5), \
        (beta3_0, beta3_1, beta3_2, beta3_3, beta3_4, beta3_5)

def compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1, delta_p2, delta_p3):
    delta_m5m1 = delta_m2 + delta_m1
    delta_m5p1 = delta_m5m1 + delta_m0
    delta_m5p3 = delta_m5p1 + delta_p1
    delta_m5p5 = delta_m5p3 + delta_p2
    delta_m5p7 = delta_m5p5 + delta_p3
    
    delta_m3p1 = delta_m1 + delta_m0
    delta_m3p3 = delta_m3p1 + delta_p1
    delta_m3p5 = delta_m3p3 + delta_p2
    delta_m3p7 = delta_m3p5 + delta_p3
    
    delta_m1p3 = delta_m0 + delta_p1
    delta_m1p5 = delta_m1p3 + delta_p2
    delta_m1p7 = delta_m1p5 + delta_p3
    
    delta_p1p5 = delta_p1 + delta_p2
    delta_p1p7 = delta_p1p5 + delta_p3
    
    delta_p3p7 = delta_p2 + delta_p3

    return (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
    (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
    (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7

def compute_polynomials3(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3):

    (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
    (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
    (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
    = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

    # POLYNOMIAL 03
    c0_0 = - ( delta_m3p1 * delta_m5p1 * delta_m2 ) / ( delta_m5m1 * delta_m0 * delta_m1 ) \
        + ( delta_m5p1 * delta_m0 ) / ( delta_m3p1 * delta_m1 ) \
        + ( delta_m2 * (delta_m3p1 * delta_m5p1 + delta_m3p1 * delta_m0 + delta_m5p1 * delta_m0) ) / ( delta_m3p1 * delta_m5p1 * delta_m0 )
    
    c0_1 = - ( delta_m3p1 * delta_m5p1 ) / ( delta_m5m1 * delta_m0 ) \
        + ( delta_m1 * (delta_m3p1 * delta_m5p1 + delta_m3p1 * delta_m0 + delta_m5p1 * delta_m0) ) / ( delta_m3p1 * delta_m5p1 * delta_m0 )
    
    c0_2 = ( (delta_m3p1 * delta_m5p1 + delta_m3p1 * delta_m0 + delta_m5p1 * delta_m0) ) / ( delta_m3p1 * delta_m5p1 )

    # POLYNOMIAL 13
    c1_0 = - ( delta_m1 * (delta_m3p1 * delta_m0 - delta_m3p1 * delta_p1 - delta_m0 * delta_p1) ) / ( delta_m3p1 * delta_m0 * delta_p1 ) \
        - (delta_m3p1 * delta_p1 ) / (delta_m1p3 * delta_m0) \
        + (delta_m3p1 * delta_m0 * delta_m1 ) / (delta_m1p3 * delta_m3p3 * delta_p1)
    
    c1_1 = - ( (delta_m3p1 * delta_m0 - delta_m3p1 * delta_p1 - delta_m0 * delta_p1) ) / (delta_m3p1 * delta_p1) \
        + (delta_m3p1 * delta_m0**2 ) / ( delta_m1p3 * delta_m3p3 * delta_p1)
    
    c1_2 = (delta_m3p1 * delta_m0 ) / (delta_m1p3 * delta_m3p3)
    
    # POLYNOMIAL 23
    c2_0 = ( (-delta_p1p5 * delta_m0 + delta_p1p5 * delta_p1 - delta_m0 * delta_p1) ) / (delta_p1p5 * delta_p1) \
        - (delta_m0**2 * delta_p1 ) / (delta_m1p5 * delta_p1p5 * delta_p2) \
        + (delta_p1p5 * delta_m0**2) / (delta_m1p3 * delta_p1 * delta_p2)
    
    c2_1 = - (delta_m0 * delta_p1**2) / (delta_m1p5 * delta_p1p5 * delta_p2) \
        + (delta_p1p5 * delta_m0) / (delta_m1p3 * delta_p2)
    
    c2_2 = - (delta_m0 * delta_p1) / (delta_m1p5 * delta_p1p5)
    
    # POLYNOMIAL 33
    c3_0 = (delta_p1p5 * delta_p1p7) / (delta_p3p7 * delta_p2) \
        + (delta_p1p5 * delta_p1**2) / (delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_p1p7 * delta_p1**2) / (delta_p1p5 * delta_p2 * delta_p3)
    
    c3_1 = (delta_p1p5 * delta_p1 * delta_p2) / (delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_p1p7 * delta_p1 ) / (delta_p1p5 * delta_p3)
    
    c3_2 = (delta_p1p5 * delta_p1 ) / (delta_p1p7 * delta_p3p7)

    return (c0_0, c0_1, c0_2), (c1_0, c1_1, c1_2), \
    (c2_0, c2_1, c2_2), (c3_0, c3_1, c3_2)

def compute_polynomial34(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3):

    (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
    (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
    (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
    = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

    c3_0 = - (delta_p1p5 * delta_p1p7 * delta_m0 - delta_p1p5 * delta_p1p7 * delta_p1 + delta_p1p5 * delta_m0 * delta_p1 + delta_p1p7 * delta_m0 * delta_p1) / (delta_p1p5 * delta_p1p7 * delta_p1) \
        + (delta_p1p5 * delta_m0**2 * delta_p1) / (delta_m1p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_p1p7 * delta_m0**2 * delta_p1) / (delta_m1p5 * delta_p1p5 * delta_p2 * delta_p3) \
        + (delta_p1p5 * delta_p1p7 * delta_m0**2) / (delta_m1p3 * delta_p3p7 * delta_p1 * delta_p2)

    c3_1 = (delta_p1p5 * delta_m0 * delta_p1**2) / (delta_m1p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_p1p7 * delta_m0 * delta_p1**2) / (delta_m1p5 * delta_p1p5 * delta_p2 * delta_p3) \
        + (delta_p1p5 * delta_p1p7 * delta_m0) / (delta_m1p3 * delta_p3p7 * delta_p2)

    c3_2 = (delta_p1p5 * delta_m0 * delta_p1 * delta_p2) / (delta_m1p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_p1p7 * delta_m0 * delta_p1) / (delta_m1p5 * delta_p1p5 * delta_p3)

    c3_3 = (delta_p1p5 * delta_m0 * delta_p1) / (delta_m1p7 * delta_p1p7 * delta_p3p7)

    return c3_0, c3_1, c3_2, c3_3

def compute_polynomial6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3):

    (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
    (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
    (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
    = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

    c6_0 = \
        - delta_m2 / delta_p1 \
        + delta_m2 / delta_m0 \
        - delta_m2 / delta_p1p7 \
        - delta_m2 / delta_p1p5 \
        + delta_m2 / delta_m5p1 \
        + delta_m2 / delta_m3p1 \
        + (delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_m0 * delta_p1) / (delta_m3p1 * delta_m3p3 * delta_m3p5 * delta_m3p7 * delta_m1) \
        + (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_m0 * delta_p1 * delta_m2) / (delta_m1p7 * delta_m3p7 * delta_m5p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_m3p1 * delta_m5p1 * delta_p1p7 * delta_m0 * delta_p1 * delta_m2) / (delta_m1p5 * delta_m3p5 * delta_m5p5 * delta_p1p5 * delta_p2 * delta_p3) \
        + (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_m0 * delta_m2) / (delta_m1p3 * delta_m3p3 * delta_m5p3 * delta_p3p7 * delta_p1 * delta_p2) \
        - (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_p1 * delta_m2) / (delta_m1p3 * delta_m1p5 * delta_m1p7 * delta_m5m1 * delta_m0 * delta_m1)
    c6_1 = \
        - delta_m1 / delta_p1 \
        + delta_m1 / delta_m0 \
        - delta_m1 / delta_p1p7 \
        - delta_m1 / delta_p1p5 \
        + delta_m1 / delta_m5p1 \
        + delta_m1 / delta_m3p1 \
        + (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_m0 * delta_p1 * delta_m1) / (delta_m1p7 * delta_m3p7 * delta_m5p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_m3p1 * delta_m5p1 * delta_p1p7 * delta_m0 * delta_p1 * delta_m1) / (delta_m1p5 * delta_m3p5 * delta_m5p5 * delta_p1p5 * delta_p2 * delta_p3) \
        + (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_m0 * delta_m1) / (delta_m1p3 * delta_m3p3 * delta_m5p3 * delta_p3p7 * delta_p1 * delta_p2) \
        - (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_p1) / (delta_m1p3 * delta_m1p5 * delta_m1p7 * delta_m5m1 * delta_m0)
    c6_2 = \
        - delta_m0 / delta_p1 \
        + 1 \
        - delta_m0 / delta_p1p7 \
        - delta_m0 / delta_p1p5 \
        + delta_m0 / delta_m5p1 \
        + delta_m0 / delta_m3p1 \
        + (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_m0**2 * delta_p1) / (delta_m1p7 * delta_m3p7 * delta_m5p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_m3p1 * delta_m5p1 * delta_p1p7 * delta_m0**2 * delta_p1) / (delta_m1p5 * delta_m3p5 * delta_m5p5 * delta_p1p5 * delta_p2 * delta_p3) \
        + (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_m0**2) / (delta_m1p3 * delta_m3p3 * delta_m5p3 * delta_p3p7 * delta_p1 * delta_p2)
    c6_3 = (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_m0 * delta_p1**2) / (delta_m1p7 * delta_m3p7 * delta_m5p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_m3p1 * delta_m5p1 * delta_p1p7 * delta_m0 * delta_p1**2) / (delta_m1p5 * delta_m3p5 * delta_m5p5 * delta_p1p5 * delta_p2 * delta_p3) \
        + (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_m0) / (delta_m1p3 * delta_m3p3 * delta_m5p3 * delta_p3p7 * delta_p2)
    c6_4 = (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_m0 * delta_p1 * delta_p2) / (delta_m1p7 * delta_m3p7 * delta_m5p7 * delta_p1p7 * delta_p3p7 * delta_p3) \
        - (delta_m3p1 * delta_m5p1 * delta_p1p7 * delta_m0 * delta_p1) / (delta_m1p5 * delta_m3p5 * delta_m5p5 * delta_p1p5 * delta_p3)
    c6_5 = (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_m0 * delta_p1) / (delta_m1p7 * delta_m3p7 * delta_m5p7 * delta_p1p7 * delta_p3p7)

    return c6_0, c6_1, c6_2, c6_3, c6_4, c6_5

def compute_smoothness_beta34(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3):

    (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
    (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
    (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
    = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

    x_m3 = delta_m2
    x_m1 = delta_m5m1
    x_p1 = delta_m5p1
    x_p3 = delta_m5p3
    x_p5 = delta_m5p5
    x_p7 = delta_m5p7

    one_beta_C0 = 1 / (delta_p1p5 * delta_p1p7 * delta_p1)
    beta_C00 = - 1.0 * one_beta_C0
    one_beta_C1 = 1.0 / (delta_m1p3 * delta_p3p7 * delta_p1 * delta_p2)
    beta_C10 = delta_m0 * one_beta_C1
    beta_C11 = delta_p1 * one_beta_C1
    one_beta_C2 = 1.0 / (delta_m1p5 * delta_p1p5 * delta_p2 * delta_p3)
    beta_C20 = - delta_m0 * one_beta_C2 
    beta_C21 = - delta_p1 * one_beta_C2
    beta_C22 = - delta_p2 * one_beta_C2
    one_beta_C3 = 1.0 / (delta_m1p7 * delta_p1p7 * delta_p3p7 * delta_p3)
    beta_C30 = delta_m0 * one_beta_C3
    beta_C31 = delta_p1 * one_beta_C3
    beta_C32 = delta_p2 * one_beta_C3
    beta_C33 = delta_p3 * one_beta_C3

    D00 = - x_m1 * x_p3 * x_p5 - x_m1 * x_p3 * x_p7 - x_m1 * x_p5 * x_p7 - x_p3 * x_p5 * x_p7
    D01 = 2 * (x_m1 * x_p3 + x_m1 * x_p5 + x_m1 * x_p7 + x_p3 * x_p5 + x_p3 * x_p7 + x_p5 * x_p7)
    D02 = - 3 * (x_m1 + x_p3 + x_p5 + x_p7)
    D03 = 4
    D10 = - x_m1 * x_p1 * x_p5 - x_m1 * x_p1 * x_p7 - x_m1 * x_p5 * x_p7 - x_p1 * x_p5 * x_p7
    D11 = 2 * (x_m1 * x_p1 + x_m1 * x_p5 + x_m1 * x_p7 + x_p1 * x_p5 + x_p1 * x_p7 + x_p5 * x_p7)
    D12 = - 3 * (x_m1 + x_p1 + x_p5 + x_p7)
    D13 = 4
    D20 = - x_m1 * x_p1 * x_p3 - x_m1 * x_p1 * x_p7 - x_m1 * x_p3 * x_p7 - x_p1 * x_p3 * x_p7
    D21 = 2 * (x_m1 * x_p1 + x_m1 * x_p3 + x_m1 * x_p7 + x_p1 * x_p3 + x_p1 * x_p7 + x_p3 * x_p7)
    D22 = - 3 * (x_m1 + x_p1 + x_p3 + x_p7)
    D23 = 4
    D30 = - x_m1 * x_p1 * x_p3 - x_m1 * x_p1 * x_p5 - x_m1 * x_p3 * x_p5 - x_p1 * x_p3 * x_p5
    D31 = 2 * (x_m1 * x_p1 + x_m1 * x_p3 + x_m1 * x_p5 + x_p1 * x_p3 + x_p1 * x_p5 + x_p3 * x_p5)
    D32 = - 3 * (x_m1 + x_p1 + x_p3 + x_p5)
    D33 = 4

    coeff_beta3_C0C0 = \
        (x_p1**5 - x_m1**5) * (9 * D03**2 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * 3 * D02 * D03 * delta_m0 \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D03 * delta_m0 + (4 * D02**2 * delta_m0) / 3 + 12 * D03**2 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D02 * delta_m0 + 12 * D02 * D03 * delta_m0**3) \
        + (x_p1 - x_m1) * (D01**2 * delta_m0 + 4 * D02**2 * delta_m0**3 + 36 * D03**2 * delta_m0**5)
    coeff_beta3_C0C1 = \
        (x_p1**5 - x_m1**5) * (18 * D03 * D13 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * (3 * D02 * D13 * delta_m0 + 3 * D03 * D12 * delta_m0) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D13 * delta_m0 + (8 * D02 * D12 * delta_m0) / 3 + 2 * D03 * D11 * delta_m0 + 24 * D03 * D13 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D12 * delta_m0 + 2 * D02 * D11 * delta_m0 + 12 * D02 * D13 * delta_m0**3 + 12 * D03 * D12 * delta_m0**3) \
        + (x_p1 - x_m1) * (2 * D01 * D11 * delta_m0 + 8 * D02 * D12 * delta_m0**3 + 72 * D03 * D13 * delta_m0**5)
    coeff_beta3_C0C2 = \
        (x_p1**5 - x_m1**5) * (18 * D03 * D23 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * (3 * D02 * D23 * delta_m0 + 3 * D03 * D22 * delta_m0) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D23 * delta_m0 + (8 * D02 * D22 * delta_m0) / 3 + 2 * D03 * D21 * delta_m0 + 24 * D03 * D23 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D22 * delta_m0 + 2 * D02 * D21 * delta_m0 + 12 * D02 * D23 * delta_m0**3 + 12 * D03 * D22 * delta_m0**3) \
        + (x_p1 - x_m1) * (2 * D01 * D21 * delta_m0 + 8 * D02 * D22 * delta_m0**3 + 72 * D03 * D23 * delta_m0**5)
    coeff_beta3_C0C3 = \
        (x_p1**5 - x_m1**5) * (18 * D03 * D33 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * (3 * D02 * D33 * delta_m0 + 3 * D03 * D32 * delta_m0) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D33 * delta_m0 + (8 * D02 * D32 * delta_m0) / 3 + 2 * D03 * D31 * delta_m0 + 24 * D03 * D33 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D32 * delta_m0 + 2 * D02 * D31 * delta_m0 + 12 * D02 * D33 * delta_m0**3 + 12 * D03 * D32 * delta_m0**3) \
        + (x_p1 - x_m1) * (2 * D01 * D31 * delta_m0 + 8 * D02 * D32 * delta_m0**3 + 72 * D03 * D33 * delta_m0**5)

    coeff_beta3_C1C1 = \
        (x_p1**5 - x_m1**5) * (9 * D13**2 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * 3 * D12 * D13 * delta_m0 \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D13 * delta_m0 + (4 * D12**2 * delta_m0) / 3 + 12 * D13**2 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D12 * delta_m0 + 12 * D12 * D13 * delta_m0**3) \
        + (x_p1 - x_m1) * (D11**2 * delta_m0 + 4 * D12**2 * delta_m0**3 + 36 * D13**2 * delta_m0**5)
    coeff_beta3_C1C2 = \
        (x_p1**5 - x_m1**5) * (18 * D13 * D23 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * (3 * D12 * D23 * delta_m0 + 3 * D13 * D22 * delta_m0) \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D23 * delta_m0 + (8 * D12 * D22 * delta_m0) / 3 + 2 * D13 * D21 * delta_m0 + 24 * D13 * D23 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D22 * delta_m0 + 2 * D12 * D21 * delta_m0 + 12 * D12 * D23 * delta_m0**3 + 12 * D13 * D22 * delta_m0**3) \
        + (x_p1 - x_m1) * (2 * D11 * D21 * delta_m0 + 8 * D12 * D22 * delta_m0**3 + 72 * D13 * D23 * delta_m0**5)
    coeff_beta3_C1C3 = \
        (x_p1**5 - x_m1**5) * (18 * D13 * D33 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * (3 * D12 * D33 * delta_m0 + 3 * D13 * D32 * delta_m0) \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D33 * delta_m0 + (8 * D12 * D32 * delta_m0) / 3 + 2 * D13 * D31 * delta_m0 + 24 * D13 * D33 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D32 * delta_m0 + 2 * D12 * D31 * delta_m0 + 12 * D12 * D33 * delta_m0**3 + 12 * D13 * D32 * delta_m0**3) \
        + (x_p1 - x_m1) * (2 * D11 * D31 * delta_m0 + 8 * D12 * D32 * delta_m0**3 + 72 * D13 * D33 * delta_m0**5)

    coeff_beta3_C2C2 = \
        (x_p1**5 - x_m1**5) * (9 * D23**2 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * 3 * D22 * D23 * delta_m0 \
        + (x_p1**3 - x_m1**3) * (2 * D21 * D23 * delta_m0 + (4 * D22**2 * delta_m0) / 3 + 12 * D23**2 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D21 * D22 * delta_m0 + 12 * D22 * D23 * delta_m0**3) \
        + (x_p1 - x_m1) * (D21**2 * delta_m0 + 4 * D22**2 * delta_m0**3 + 36 * D23**2 * delta_m0**5)
    coeff_beta3_C2C3 = \
        (x_p1**5 - x_m1**5) * (18 * D23 * D33 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * (3 * D22 * D33 * delta_m0 + 3 * D23 * D32 * delta_m0) \
        + (x_p1**3 - x_m1**3) * (2 * D21 * D33 * delta_m0 + (8 * D22 * D32 * delta_m0) / 3 + 2 * D23 * D31 * delta_m0 + 24 * D23 * D33 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D21 * D32 * delta_m0 + 2 * D22 * D31 * delta_m0 + 12 * D22 * D33 * delta_m0**3 + 12 * D23 * D32 * delta_m0**3) \
        + (x_p1 - x_m1) * (2 * D21 * D31 * delta_m0 + 8 * D22 * D32 * delta_m0**3 + 72 * D23 * D33 * delta_m0**5)

    coeff_beta3_C3C3 = \
        (x_p1**5 - x_m1**5) * (9 * D33**2 * delta_m0) / 5 \
        + (x_p1**4 - x_m1**4) * 3 * D32 * D33 * delta_m0 \
        + (x_p1**3 - x_m1**3) * (2 * D31 * D33 * delta_m0 + (4 * D32**2 * delta_m0) / 3 + 12 * D33**2 * delta_m0**3) \
        + (x_p1**2 - x_m1**2) * (2 * D31 * D32 * delta_m0 + 12 * D32 * D33 * delta_m0**3) \
        + (x_p1 - x_m1) * (D31**2 * delta_m0 + 4 * D32**2 * delta_m0**3 + 36 * D33**2 * delta_m0**5)

    return (beta_C00, beta_C10, beta_C11, beta_C20, beta_C21, beta_C22,
            beta_C30, beta_C31, beta_C32, beta_C33), \
            (coeff_beta3_C0C0, coeff_beta3_C0C1, coeff_beta3_C0C2, coeff_beta3_C0C3,
             coeff_beta3_C1C1, coeff_beta3_C1C2, coeff_beta3_C1C3,
             coeff_beta3_C2C2, coeff_beta3_C2C3,
             coeff_beta3_C3C3)

def compute_smoothness_beta6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3):

    (delta_m5m1, delta_m5p1, delta_m5p3, delta_m5p5, delta_m5p7), \
    (delta_m3p1, delta_m3p3, delta_m3p5, delta_m3p7), \
    (delta_m1p3, delta_m1p5, delta_m1p7), (delta_p1p5, delta_p1p7), delta_p3p7 \
    = compute_cumulative_deltas6(delta_m2, delta_m1, delta_m0, delta_p1 , delta_p2, delta_p3)

    x_m3 = delta_m2
    x_m1 = delta_m5m1
    x_p1 = delta_m5p1
    x_p3 = delta_m5p3
    x_p5 = delta_m5p5
    x_p7 = delta_m5p7

    one_C0 = 1.0 / (delta_m3p1 * delta_m3p3 * delta_m3p5 * delta_m3p7 * delta_m1)
    C00 = - 1.0 * one_C0        # u_{i-2}
    one_C1 = 1.0 / (delta_m1p3 * delta_m1p5 * delta_m1p7 * delta_m5m1 * delta_m0 * delta_m1)
    C10 = delta_m2 * one_C1     # u_{i-2}
    C11 = delta_m1 * one_C1     # u_{i-1}
    one_C2 = 1.0 / (delta_m3p1 * delta_m5p1 * delta_p1p5 * delta_p1p7 * delta_m0 * delta_p1)
    C20 = - delta_m2 * one_C2   # u_{i-2}
    C21 = - delta_m1 * one_C2   # u_{i-1}
    C22 = - delta_m0 * one_C2   # u_{i}
    one_C3 = 1.0 / (delta_m1p3 * delta_m3p3 * delta_m5p3 * delta_p3p7 * delta_p1 * delta_p2)
    C30 = delta_m2 * one_C3 # u_{i-2}
    C31 = delta_m1 * one_C3 # u_{i-1}
    C32 = delta_m0 * one_C3 # u_{i}
    C33 = delta_p1 * one_C3 # u_{i+1}
    one_C4 = 1.0 / (delta_m1p5 * delta_m3p5 * delta_m5p5 * delta_p1p5 * delta_p2 * delta_p3)
    C40 = - delta_m2 * one_C4   # u_{i-2}
    C41 = - delta_m1 * one_C4   # u_{i-1}
    C42 = - delta_m0 * one_C4   # u_{i}
    C43 = - delta_p1 * one_C4   # u_{i+1}
    C44 = - delta_p2 * one_C4   # u_{i+2}
    one_C5 = 1.0 / (delta_m1p7 * delta_m3p7 * delta_m5p7 * delta_p1p7 * delta_p3p7 * delta_p3)
    C50 = delta_m2 * one_C5     # u_{i-2}
    C51 = delta_m1 * one_C5     # u_{i-1}
    C52 = delta_m0 * one_C5     # u_{i}
    C53 = delta_p1 * one_C5     # u_{i+1}
    C54 = delta_p2 * one_C5     # u_{i+2}
    C55 = delta_p3 * one_C5     # u_{i+3}

    D01 = 2.0 * (x_m1 * x_p1 * x_p3 * x_p5 + x_m1 * x_p1 * x_p3 * x_p7 + x_m1 * x_p1 * x_p5 * x_p7 + x_m1 * x_p3 * x_p5 * x_p7 + x_p1 * x_p3 * x_p5 * x_p7)
    D02 = -3.0 * (x_m1 * x_p1 * x_p3 + x_m1 * x_p1 * x_p5 + x_m1 * x_p1 * x_p7 + x_m1 * x_p3 * x_p5 + x_m1 * x_p3 * x_p7 + x_m1 * x_p5 * x_p7 + x_p1 * x_p3 * x_p5 + x_p1 * x_p3 * x_p7 + x_p1 * x_p5 * x_p7 + x_p3 * x_p5 * x_p7)
    D03 = 4.0 * (x_m1 * x_p1 + x_m1 * x_p3 + x_m1 * x_p5 + x_m1 * x_p7 + x_p1 * x_p3 + x_p1 * x_p5 + x_p1 * x_p7 + x_p3 * x_p5 + x_p3 * x_p7 + x_p5 * x_p7)
    D04 = -5.0 * (x_m1 + x_p1 + x_p3 + x_p5 + x_p7)
    D05 = 6.0
    D11 = 2.0 * (x_m3 * x_p1 * x_p3 * x_p5 + x_m3 * x_p1 * x_p3 * x_p7 + x_m3 * x_p1 * x_p5 * x_p7 + x_m3 * x_p3 * x_p5 * x_p7 + x_p1 * x_p3 * x_p5 * x_p7)
    D12 = -3.0 * (x_m3 * x_p1 * x_p3 + x_m3 * x_p1 * x_p5 + x_m3 * x_p1 * x_p7 + x_m3 * x_p3 * x_p5 + x_m3 * x_p3 * x_p7 + x_m3 * x_p5 * x_p7 + x_p1 * x_p3 * x_p5 + x_p1 * x_p3 * x_p7 + x_p1 * x_p5 * x_p7 + x_p3 * x_p5 * x_p7)
    D13 = 4.0 * (x_m3 * x_p1 + x_m3 * x_p3 + x_m3 * x_p5 + x_m3 * x_p7 + x_p1 * x_p3 + x_p1 * x_p5 + x_p1 * x_p7 + x_p3 * x_p5 + x_p3 * x_p7 + x_p5 * x_p7)
    D14 = -5.0 * (x_m3 + x_p1 + x_p3 + x_p5 + x_p7)
    D15 = 6.0
    D21 = 2.0 * (x_m1 * x_m3 * x_p3 * x_p5 + x_m1 * x_m3 * x_p3 * x_p7 + x_m1 * x_m3 * x_p5 * x_p7 + x_m1 * x_p3 * x_p5 * x_p7 + x_m3 * x_p3 * x_p5 * x_p7)
    D22 = -3.0 * (x_m1 * x_m3 * x_p3 + x_m1 * x_m3 * x_p5 + x_m1 * x_m3 * x_p7 + x_m1 * x_p3 * x_p5 + x_m1 * x_p3 * x_p7 + x_m1 * x_p5 * x_p7 + x_m3 * x_p3 * x_p5 + x_m3 * x_p3 * x_p7 + x_m3 * x_p5 * x_p7 + x_p3 * x_p5 * x_p7)
    D23 = 4.0 * (x_m1 * x_m3 + x_m1 * x_p3 + x_m1 * x_p5 + x_m1 * x_p7 + x_m3 * x_p3 + x_m3 * x_p5 + x_m3 * x_p7 + x_p3 * x_p5 + x_p3 * x_p7 + x_p5 * x_p7)
    D24 = -5.0 * (x_m1 + x_m3 + x_p3 + x_p5 + x_p7)
    D25 = 6.0
    D31 = 2.0 * (x_m1 * x_m3 * x_p1 * x_p5 + x_m1 * x_m3 * x_p1 * x_p7 + x_m1 * x_m3 * x_p5 * x_p7 + x_m1 * x_p1 * x_p5 * x_p7 + x_m3 * x_p1 * x_p5 * x_p7)
    D32 = -3.0 * (x_m1 * x_m3 * x_p1 + x_m1 * x_m3 * x_p5 + x_m1 * x_m3 * x_p7 + x_m1 * x_p1 * x_p5 + x_m1 * x_p1 * x_p7 + x_m1 * x_p5 * x_p7 + x_m3 * x_p1 * x_p5 + x_m3 * x_p1 * x_p7 + x_m3 * x_p5 * x_p7 + x_p1 * x_p5 * x_p7)
    D33 = 4.0 * (x_m1 * x_m3 + x_m1 * x_p1 + x_m1 * x_p5 + x_m1 * x_p7 + x_m3 * x_p1 + x_m3 * x_p5 + x_m3 * x_p7 + x_p1 * x_p5 + x_p1 * x_p7 + x_p5 * x_p7)
    D34 = -5.0 * (x_m1 + x_m3 + x_p1 + x_p5 + x_p7)
    D35 = 6.0
    D41 = 2.0 * (x_m1 * x_m3 * x_p1 * x_p3 + x_m1 * x_m3 * x_p1 * x_p7 + x_m1 * x_m3 * x_p3 * x_p7 + x_m1 * x_p1 * x_p3 * x_p7 + x_m3 * x_p1 * x_p3 * x_p7)
    D42 = -3.0 * (x_m1 * x_m3 * x_p1 + x_m1 * x_m3 * x_p3 + x_m1 * x_m3 * x_p7 + x_m1 * x_p1 * x_p3 + x_m1 * x_p1 * x_p7 + x_m1 * x_p3 * x_p7 + x_m3 * x_p1 * x_p3 + x_m3 * x_p1 * x_p7 + x_m3 * x_p3 * x_p7 + x_p1 * x_p3 * x_p7)
    D43 = 4.0 * (x_m1 * x_m3 + x_m1 * x_p1 + x_m1 * x_p3 + x_m1 * x_p7 + x_m3 * x_p1 + x_m3 * x_p3 + x_m3 * x_p7 + x_p1 * x_p3 + x_p1 * x_p7 + x_p3 * x_p7)
    D44 = -5.0 * (x_m1 + x_m3 + x_p1 + x_p3 + x_p7)
    D45 = 6.0
    D51 = 2.0 * (x_m1 * x_m3 * x_p1 * x_p3 + x_m1 * x_m3 * x_p1 * x_p5 + x_m1 * x_m3 * x_p3 * x_p5 + x_m1 * x_p1 * x_p3 * x_p5 + x_m3 * x_p1 * x_p3 * x_p5)
    D52 = -3.0 * (x_m1 * x_m3 * x_p1 + x_m1 * x_m3 * x_p3 + x_m1 * x_m3 * x_p5 + x_m1 * x_p1 * x_p3 + x_m1 * x_p1 * x_p5 + x_m1 * x_p3 * x_p5 + x_m3 * x_p1 * x_p3 + x_m3 * x_p1 * x_p5 + x_m3 * x_p3 * x_p5 + x_p1 * x_p3 * x_p5)
    D53 = 4.0 * (x_m1 * x_m3 + x_m1 * x_p1 + x_m1 * x_p3 + x_m1 * x_p5 + x_m3 * x_p1 + x_m3 * x_p3 + x_m3 * x_p5 + x_p1 * x_p3 + x_p1 * x_p5 + x_p3 * x_p5)
    D54 = -5.0 * (x_m1 + x_m3 + x_p1 + x_p3 + x_p5)
    D55 = 6.0

    coeff_C0C0 = \
        (x_p1**9 - x_m1**9) * (25 * D05**2 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D04 * D05 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D03 * D05 * delta_m0) / 7 + (16 * D04**2 * delta_m0) / 7 + (400 * D05**2 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D02 * D05 * delta_m0) / 3 + 4 * D03 * D04 * delta_m0 + 80 * D04 * D05 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D01 * D05 * delta_m0 + (16 * D02 * D04 * delta_m0) / 5 + (9 * D03**2 * delta_m0) / 5 + 48 * D03 * D05 * delta_m0**3 + (144 * D04**2 * delta_m0**3) / 5 + 720 * D05**2 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D01 * D04 * delta_m0 + 3 * D02 * D03 * delta_m0 + 20 * D02 * D05 * delta_m0**3 + 36 * D03 * D04 * delta_m0**3 + 720 * D04 * D05 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D03 * delta_m0 + (4 * D02**2 * delta_m0) / 3 + 16 * D02 * D04 * delta_m0**3 + 12 * D03**2 * delta_m0**3 + 240 * D03 * D05 * delta_m0**5 + 192 * D04**2 * delta_m0**5 + 4800 * D05**2 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D02 * delta_m0 + 12 * D02 * D03 * delta_m0**3 + 144 * D03 * D04 * delta_m0**5 + 2880 * D04 * D05 * delta_m0**7) \
        + (x_p1 - x_m1) * (D01**2 * delta_m0 + 4 * D02**2 * delta_m0**3 + 36 * D03**2 * delta_m0**5 + 576 * D04**2 * delta_m0**7 + 14400 * D05**2 * delta_m0**9)
    coeff_C0C1 = \
        (x_p1**9 - x_m1**9) * (50* D05 * D15 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5* D04 * D15 * delta_m0 + 5* D05 * D14 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30* D03 * D15 * delta_m0) / 7 + (32 * D04 * D14 * delta_m0) / 7 + (30* D05 * D13 * delta_m0) / 7 + (800* D05 * D15 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10* D02 * D15 * delta_m0) / 3 + 4* D03 * D14 * delta_m0 + 4* D04 * D13 * delta_m0 + 80* D04 * D15 * delta_m0**3 + (10* D05 * D12 * delta_m0) / 3 + 80* D05 * D14 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D01 * D15 * delta_m0 + (16* D02 * D14 * delta_m0) / 5 + (18 * D03 * D13 * delta_m0) / 5 + 48 * D03 * D15 * delta_m0**3 + (16* D04 * D12 * delta_m0) / 5 + (288 * D04 * D14 * delta_m0**3) / 5 + 2 * D05 * D11 * delta_m0 + 48 * D05 * D13 * delta_m0**3 + 1440* D05 * D15 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D01 * D14 * delta_m0 + 3* D02 * D13 * delta_m0 + 20* D02 * D15 * delta_m0**3 + 3* D03 * D12 * delta_m0 + 36* D03 * D14 * delta_m0**3 + 2 * D04 * D11 * delta_m0 + 36* D04 * D13 * delta_m0**3 + 720* D04 * D15 * delta_m0**5 + 20* D05 * D12 * delta_m0**3 + 720* D05 * D14 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D13 * delta_m0 + (8 * D02 * D12 * delta_m0) / 3 + 16* D02 * D14 * delta_m0**3 + 2 * D03 * D11 * delta_m0 + 24* D03 * D13 * delta_m0**3 + 240* D03 * D15 * delta_m0**5 + 16* D04 * D12 * delta_m0**3 + 384* D04 * D14 * delta_m0**5 + 240* D05 * D13 * delta_m0**5 + 9600* D05 * D15 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D12 * delta_m0 + 2 * D02 * D11 * delta_m0 + 12 * D02 * D13 * delta_m0**3 + 12 * D03 * D12 * delta_m0**3 + 144* D03 * D14 * delta_m0**5 + 144* D04 * D13 * delta_m0**5 + 2880* D04 * D15 * delta_m0**7 + 2880* D05 * D14 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D01 * D11 * delta_m0 + 8 * D02 * D12 * delta_m0**3 + 72 * D03 * D13 * delta_m0**5 + 1152 * D04 * D14 * delta_m0**7 + 28800* D05 * D15 * delta_m0**9)
    coeff_C0C2 = \
        (x_p1**9 - x_m1**9) * (50 * D05 * D25 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D04 * D25 * delta_m0 + 5* D05 * D24 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D03 * D25 * delta_m0) / 7 + (32 * D04 * D24 * delta_m0) / 7 + (30* D05 * D23 * delta_m0) / 7 + (800* D05 * D25 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D02 * D25 * delta_m0) / 3 + 4* D03 * D24 * delta_m0 + 4* D04 * D23 * delta_m0 + 80* D04 * D25 * delta_m0**3 + (10* D05 * D22 * delta_m0) / 3 + 80* D05 * D24 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D01 * D25 * delta_m0 + (16* D02 * D24 * delta_m0) / 5 + (18 * D03 * D23 * delta_m0) / 5 + 48 * D03 * D25 * delta_m0**3 + (16* D04 * D22 * delta_m0) / 5 + (288 * D04 * D24 * delta_m0**3) / 5 + 2 * D05 * D21 * delta_m0 + 48 * D05 * D23 * delta_m0**3 + 1440* D05 * D25 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D01 * D24 * delta_m0 + 3* D02 * D23 * delta_m0 + 20* D02 * D25 * delta_m0**3 + 3* D03 * D22 * delta_m0 + 36* D03 * D24 * delta_m0**3 + 2 * D04 * D21 * delta_m0 + 36* D04 * D23 * delta_m0**3 + 720* D04 * D25 * delta_m0**5 + 20* D05 * D22 * delta_m0**3 + 720* D05 * D24 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D23 * delta_m0 + (8 * D02 * D22 * delta_m0) / 3 + 16* D02 * D24 * delta_m0**3 + 2 * D03 * D21 * delta_m0 + 24* D03 * D23 * delta_m0**3 + 240* D03 * D25 * delta_m0**5 + 16* D04 * D22 * delta_m0**3 + 384* D04 * D24 * delta_m0**5 + 240* D05 * D23 * delta_m0**5 + 9600* D05 * D25 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D22 * delta_m0 + 2 * D02 * D21 * delta_m0 + 12 * D02 * D23 * delta_m0**3 + 12 * D03 * D22 * delta_m0**3 + 144* D03 * D24 * delta_m0**5 + 144* D04 * D23 * delta_m0**5 + 2880* D04 * D25 * delta_m0**7 + 2880* D05 * D24 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D01 * D21 * delta_m0 + 8 * D02 * D22 * delta_m0**3 + 72 * D03 * D23 * delta_m0**5 + 1152 * D04 * D24 * delta_m0**7 + 28800* D05 * D25 * delta_m0**9)
    coeff_C0C3 = \
        (x_p1**9 - x_m1**9) * (50 * D05 * D35 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D04 * D35 * delta_m0 + 5* D05 * D34 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D03 * D35 * delta_m0) / 7 + (32 * D04 * D34 * delta_m0) / 7 + (30* D05 * D33 * delta_m0) / 7 + (800* D05 * D35 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D02 * D35 * delta_m0) / 3 + 4* D03 * D34 * delta_m0 + 4* D04 * D33 * delta_m0 + 80* D04 * D35 * delta_m0**3 + (10* D05 * D32 * delta_m0) / 3 + 80* D05 * D34 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D01 * D35 * delta_m0 + (16* D02 * D34 * delta_m0) / 5 + (18 * D03 * D33 * delta_m0) / 5 + 48 * D03 * D35 * delta_m0**3 + (16* D04 * D32 * delta_m0) / 5 + (288 * D04 * D34 * delta_m0**3) / 5 + 2 * D05 * D31 * delta_m0 + 48 * D05 * D33 * delta_m0**3 + 1440* D05 * D35 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D01 * D34 * delta_m0 + 3* D02 * D33 * delta_m0 + 20* D02 * D35 * delta_m0**3 + 3* D03 * D32 * delta_m0 + 36* D03 * D34 * delta_m0**3 + 2 * D04 * D31 * delta_m0 + 36* D04 * D33 * delta_m0**3 + 720* D04 * D35 * delta_m0**5 + 20* D05 * D32 * delta_m0**3 + 720* D05 * D34 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D33 * delta_m0 + (8 * D02 * D32 * delta_m0) / 3 + 16* D02 * D34 * delta_m0**3 + 2 * D03 * D31 * delta_m0 + 24* D03 * D33 * delta_m0**3 + 240* D03 * D35 * delta_m0**5 + 16* D04 * D32 * delta_m0**3 + 384* D04 * D34 * delta_m0**5 + 240* D05 * D33 * delta_m0**5 + 9600* D05 * D35 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D32 * delta_m0 + 2 * D02 * D31 * delta_m0 + 12 * D02 * D33 * delta_m0**3 + 12 * D03 * D32 * delta_m0**3 + 144* D03 * D34 * delta_m0**5 + 144* D04 * D33 * delta_m0**5 + 2880* D04 * D35 * delta_m0**7 + 2880* D05 * D34 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D01 * D31 * delta_m0 + 8 * D02 * D32 * delta_m0**3 + 72 * D03 * D33 * delta_m0**5 + 1152 * D04 * D34 * delta_m0**7 + 28800* D05 * D35 * delta_m0**9)
    coeff_C0C4 = \
        (x_p1**9 - x_m1**9) * (50* D05 * D45 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5* D04 * D45 * delta_m0 + 5* D05 * D44 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30* D03 * D45 * delta_m0) / 7 + (32 * D04 * D44 * delta_m0) / 7 + (30* D05 * D43 * delta_m0) / 7 + (800* D05 * D45 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10* D02 * D45 * delta_m0) / 3 + 4* D03 * D44 * delta_m0 + 4* D04 * D43 * delta_m0 + 80* D04 * D45 * delta_m0**3 + (10* D05 * D42 * delta_m0) / 3 + 80* D05 * D44 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D01 * D45 * delta_m0 + (16* D02 * D44 * delta_m0) / 5 + (18 * D03 * D43 * delta_m0) / 5 + 48 * D03 * D45 * delta_m0**3 + (16* D04 * D42 * delta_m0) / 5 + (288 * D04 * D44 * delta_m0**3) / 5 + 2 * D05 * D41 * delta_m0 + 48 * D05 * D43 * delta_m0**3 + 1440* D05 * D45 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D01 * D44 * delta_m0 + 3* D02 * D43 * delta_m0 + 20* D02 * D45 * delta_m0**3 + 3* D03 * D42 * delta_m0 + 36* D03 * D44 * delta_m0**3 + 2 * D04 * D41 * delta_m0 + 36* D04 * D43 * delta_m0**3 + 720* D04 * D45 * delta_m0**5 + 20* D05 * D42 * delta_m0**3 + 720* D05 * D44 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D43 * delta_m0 + (8 * D02 * D42 * delta_m0) / 3 + 16* D02 * D44 * delta_m0**3 + 2 * D03 * D41 * delta_m0 + 24* D03 * D43 * delta_m0**3 + 240* D03 * D45 * delta_m0**5 + 16* D04 * D42 * delta_m0**3 + 384* D04 * D44 * delta_m0**5 + 240* D05 * D43 * delta_m0**5 + 9600* D05 * D45 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D42 * delta_m0 + 2 * D02 * D41 * delta_m0 + 12 * D02 * D43 * delta_m0**3 + 12 * D03 * D42 * delta_m0**3 + 144* D03 * D44 * delta_m0**5 + 144* D04 * D43 * delta_m0**5 + 2880* D04 * D45 * delta_m0**7 + 2880* D05 * D44 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D01 * D41 * delta_m0 + 8 * D02 * D42 * delta_m0**3 + 72 * D03 * D43 * delta_m0**5 + 1152 * D04 * D44 * delta_m0**7 + 28800* D05 * D45 * delta_m0**9)
    coeff_C0C5 = \
        (x_p1**9 - x_m1**9) * (50* D05 * D55 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5* D04 * D55 * delta_m0 + 5* D05 * D54 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30* D03 * D55 * delta_m0) / 7 + (32 * D04 * D54 * delta_m0) / 7 + (30* D05 * D53 * delta_m0) / 7 + (800* D05 * D55 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10* D02 * D55 * delta_m0) / 3 + 4* D03 * D54 * delta_m0 + 4* D04 * D53 * delta_m0 + 80* D04 * D55 * delta_m0**3 + (10* D05 * D52 * delta_m0) / 3 + 80* D05 * D54 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D01 * D55 * delta_m0 + (16* D02 * D54 * delta_m0) / 5 + (18 * D03 * D53 * delta_m0) / 5 + 48 * D03 * D55 * delta_m0**3 + (16* D04 * D52 * delta_m0) / 5 + (288 * D04 * D54 * delta_m0**3) / 5 + 2 * D05 * D51 * delta_m0 + 48 * D05 * D53 * delta_m0**3 + 1440* D05 * D55 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D01 * D54 * delta_m0 + 3* D02 * D53 * delta_m0 + 20* D02 * D55 * delta_m0**3 + 3* D03 * D52 * delta_m0 + 36* D03 * D54 * delta_m0**3 + 2 * D04 * D51 * delta_m0 + 36* D04 * D53 * delta_m0**3 + 720* D04 * D55 * delta_m0**5 + 20* D05 * D52 * delta_m0**3 + 720* D05 * D54 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D01 * D53 * delta_m0 + (8 * D02 * D52 * delta_m0) / 3 + 16* D02 * D54 * delta_m0**3 + 2 * D03 * D51 * delta_m0 + 24* D03 * D53 * delta_m0**3 + 240* D03 * D55 * delta_m0**5 + 16* D04 * D52 * delta_m0**3 + 384* D04 * D54 * delta_m0**5 + 240* D05 * D53 * delta_m0**5 + 9600* D05 * D55 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D01 * D52 * delta_m0 + 2 * D02 * D51 * delta_m0 + 12 * D02 * D53 * delta_m0**3 + 12 * D03 * D52 * delta_m0**3 + 144* D03 * D54 * delta_m0**5 + 144* D04 * D53 * delta_m0**5 + 2880* D04 * D55 * delta_m0**7 + 2880* D05 * D54 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D01 * D51 * delta_m0 + 8 * D02 * D52 * delta_m0**3 + 72 * D03 * D53 * delta_m0**5 + 1152 * D04 * D54 * delta_m0**7 + 28800* D05 * D55 * delta_m0**9)

    coeff_C1C1 = \
        (x_p1**9 - x_m1**9) * (25 * D15**2 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D14 * D15 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D13 * D15 * delta_m0) / 7 + (16 * D14**2 * delta_m0) / 7 + (400 * D15**2 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D12 * D15 * delta_m0) / 3 + 4 * D13 * D14 * delta_m0 + 80 * D14 * D15 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D11 * D15 * delta_m0 + (16 * D12 * D14 * delta_m0) / 5 + (9 * D13**2 * delta_m0) / 5 + 48 * D13 * D15 * delta_m0**3 + (144 * D14**2 * delta_m0**3) / 5 + 720 * D15**2 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D11 * D14 * delta_m0 + 3 * D12 * D13 * delta_m0 + 20 * D12 * D15 * delta_m0**3 + 36 * D13 * D14 * delta_m0**3 + 720 * D14 * D15 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D13 * delta_m0 + (4 * D12**2 * delta_m0) / 3 + 16 * D12 * D14 * delta_m0**3 + 12 * D13**2 * delta_m0**3 + 240 * D13 * D15 * delta_m0**5 + 192 * D14**2 * delta_m0**5 + 4800 * D15**2 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D12 * delta_m0 + 12 * D12 * D13 * delta_m0**3 + 144 * D13 * D14 * delta_m0**5 + 2880 * D14 * D15 * delta_m0**7) \
        + (x_p1 - x_m1) * ( D11**2 * delta_m0 + 4 * D12**2 * delta_m0**3 + 36 * D13**2 * delta_m0**5 + 576 * D14**2 * delta_m0**7 + 14400 * D15**2 * delta_m0**9)
    coeff_C1C2 = \
        (x_p1**9 - x_m1**9) * (50 * D15 * D25 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D14 * D25 * delta_m0 + 5 * D15 * D24 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D13 * D25 * delta_m0) / 7 + (32 * D14 * D24 * delta_m0) / 7 + (30 * D15 * D23 * delta_m0) / 7 + (800 * D15 * D25 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D12 * D25 * delta_m0) / 3 + 4 * D13 * D24 * delta_m0 + 4 * D14 * D23 * delta_m0 + 80 * D14 * D25 * delta_m0**3 + (10 * D15 * D22 * delta_m0) / 3 + 80 * D15 * D24 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D11 * D25 * delta_m0 + (16 * D12 * D24 * delta_m0) / 5 + (18 * D13 * D23 * delta_m0) / 5 + 48 * D13 * D25 * delta_m0**3 + (16 * D14 * D22 * delta_m0) / 5 + (288 * D14 * D24 * delta_m0**3) / 5 + 2 * D15 * D21 * delta_m0 + 48 * D15 * D23 * delta_m0**3 + 1440 * D15 * D25 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D11 * D24 * delta_m0 + 3 * D12 * D23 * delta_m0 + 20 * D12 * D25 * delta_m0**3 + 3 * D13 * D22 * delta_m0 + 36 * D13 * D24 * delta_m0**3 + 2 * D14 * D21 * delta_m0 + 36 * D14 * D23 * delta_m0**3 + 720 * D14 * D25 * delta_m0**5 + 20 * D15 * D22 * delta_m0**3 + 720 * D15 * D24 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D23 * delta_m0 + (8 * D12 * D22 * delta_m0) / 3 + 16 * D12 * D24 * delta_m0**3 + 2 * D13 * D21 * delta_m0 + 24 * D13 * D23 * delta_m0**3 + 240 * D13 * D25 * delta_m0**5 + 16 * D14 * D22 * delta_m0**3 + 384 * D14 * D24 * delta_m0**5 + 240 * D15 * D23 * delta_m0**5 + 9600 * D15 * D25 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D22 * delta_m0 + 2 * D12 * D21 * delta_m0 + 12 * D12 * D23 * delta_m0**3 + 12 * D13 * D22 * delta_m0**3 + 144 * D13 * D24 * delta_m0**5 + 144 * D14 * D23 * delta_m0**5 + 2880 * D14 * D25 * delta_m0**7 + 2880 * D15 * D24 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D11 * D21 * delta_m0 + 8 * D12 * D22 * delta_m0**3 + 72 * D13 * D23 * delta_m0**5 + 1152 * D14 * D24 * delta_m0**7 + 28800 * D15 * D25 * delta_m0**9)
    coeff_C1C3 = \
        (x_p1**9 - x_m1**9) * (50 * D15 * D35 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D14 * D35 * delta_m0 + 5 * D15 * D34 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D13 * D35 * delta_m0) / 7 + (32 * D14 * D34 * delta_m0) / 7 + (30 * D15 * D33 * delta_m0) / 7 + (800 * D15 * D35 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D12 * D35 * delta_m0) / 3 + 4 * D13 * D34 * delta_m0 + 4 * D14 * D33 * delta_m0 + 80 * D14 * D35 * delta_m0**3 + (10 * D15 * D32 * delta_m0) / 3 + 80 * D15 * D34 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D11 * D35 * delta_m0 + (16 * D12 * D34 * delta_m0) / 5 + (18 * D13 * D33 * delta_m0) / 5 + 48 * D13 * D35 * delta_m0**3 + (16 * D14 * D32 * delta_m0) / 5 + (288 * D14 * D34 * delta_m0**3) / 5 + 2 * D15 * D31 * delta_m0 + 48 * D15 * D33 * delta_m0**3 + 1440 * D15 * D35 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D11 * D34 * delta_m0 + 3 * D12 * D33 * delta_m0 + 20 * D12 * D35 * delta_m0**3 + 3 * D13 * D32 * delta_m0 + 36 * D13 * D34 * delta_m0**3 + 2 * D14 * D31 * delta_m0 + 36 * D14 * D33 * delta_m0**3 + 720 * D14 * D35 * delta_m0**5 + 20 * D15 * D32 * delta_m0**3 + 720 * D15 * D34 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D33 * delta_m0 + (8 * D12 * D32 * delta_m0) / 3 + 16 * D12 * D34 * delta_m0**3 + 2 * D13 * D31 * delta_m0 + 24 * D13 * D33 * delta_m0**3 + 240 * D13 * D35 * delta_m0**5 + 16 * D14 * D32 * delta_m0**3 + 384 * D14 * D34 * delta_m0**5 + 240 * D15 * D33 * delta_m0**5 + 9600 * D15 * D35 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D32 * delta_m0 + 2 * D12 * D31 * delta_m0 + 12 * D12 * D33 * delta_m0**3 + 12 * D13 * D32 * delta_m0**3 + 144 * D13 * D34 * delta_m0**5 + 144 * D14 * D33 * delta_m0**5 + 2880 * D14 * D35 * delta_m0**7 + 2880 * D15 * D34 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D11 * D31 * delta_m0 + 8 * D12 * D32 * delta_m0**3 + 72 * D13 * D33 * delta_m0**5 + 1152 * D14 * D34 * delta_m0**7 + 28800 * D15 * D35 * delta_m0**9)
    coeff_C1C4 = \
        (x_p1**9 - x_m1**9) * (50 * D15 * D45 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D14 * D45 * delta_m0 + 5 * D15 * D44 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D13 * D45 * delta_m0) / 7 + (32 * D14 * D44 * delta_m0) / 7 + (30 * D15 * D43 * delta_m0) / 7 + (800 * D15 * D45 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D12 * D45 * delta_m0) / 3 + 4 * D13 * D44 * delta_m0 + 4 * D14 * D43 * delta_m0 + 80 * D14 * D45 * delta_m0**3 + (10 * D15 * D42 * delta_m0) / 3 + 80 * D15 * D44 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D11 * D45 * delta_m0 + (16 * D12 * D44 * delta_m0) / 5 + (18 * D13 * D43 * delta_m0) / 5 + 48 * D13 * D45 * delta_m0**3 + (16 * D14 * D42 * delta_m0) / 5 + (288 * D14 * D44 * delta_m0**3) / 5 + 2 * D15 * D41 * delta_m0 + 48 * D15 * D43 * delta_m0**3 + 1440 * D15 * D45 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D11 * D44 * delta_m0 + 3 * D12 * D43 * delta_m0 + 20 * D12 * D45 * delta_m0**3 + 3 * D13 * D42 * delta_m0 + 36 * D13 * D44 * delta_m0**3 + 2 * D14 * D41 * delta_m0 + 36 * D14 * D43 * delta_m0**3 + 720 * D14 * D45 * delta_m0**5 + 20 * D15 * D42 * delta_m0**3 + 720 * D15 * D44 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D43 * delta_m0 + (8 * D12 * D42 * delta_m0) / 3 + 16 * D12 * D44 * delta_m0**3 + 2 * D13 * D41 * delta_m0 + 24 * D13 * D43 * delta_m0**3 + 240 * D13 * D45 * delta_m0**5 + 16 * D14 * D42 * delta_m0**3 + 384 * D14 * D44 * delta_m0**5 + 240 * D15 * D43 * delta_m0**5 + 9600 * D15 * D45 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D42 * delta_m0 + 2 * D12 * D41 * delta_m0 + 12 * D12 * D43 * delta_m0**3 + 12 * D13 * D42 * delta_m0**3 + 144 * D13 * D44 * delta_m0**5 + 144 * D14 * D43 * delta_m0**5 + 2880 * D14 * D45 * delta_m0**7 + 2880 * D15 * D44 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D11 * D41 * delta_m0 + 8 * D12 * D42 * delta_m0**3 + 72 * D13 * D43 * delta_m0**5 + 1152 * D14 * D44 * delta_m0**7 + 28800 * D15 * D45 * delta_m0**9)
    coeff_C1C5 = \
        (x_p1**9 - x_m1**9) * (50 * D15 * D55 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D14 * D55 * delta_m0 + 5 * D15 * D54 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D13 * D55 * delta_m0) / 7 + (32 * D14 * D54 * delta_m0) / 7 + (30 * D15 * D53 * delta_m0) / 7 + (800 * D15 * D55 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D12 * D55 * delta_m0) / 3 + 4 * D13 * D54 * delta_m0 + 4 * D14 * D53 * delta_m0 + 80 * D14 * D55 * delta_m0**3 + (10 * D15 * D52 * delta_m0) / 3 + 80 * D15 * D54 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D11 * D55 * delta_m0 + (16 * D12 * D54 * delta_m0) / 5 + (18 * D13 * D53 * delta_m0) / 5 + 48 * D13 * D55 * delta_m0**3 + (16 * D14 * D52 * delta_m0) / 5 + (288 * D14 * D54 * delta_m0**3) / 5 + 2 * D15 * D51 * delta_m0 + 48 * D15 * D53 * delta_m0**3 + 1440 * D15 * D55 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D11 * D54 * delta_m0 + 3 * D12 * D53 * delta_m0 + 20 * D12 * D55 * delta_m0**3 + 3 * D13 * D52 * delta_m0 + 36 * D13 * D54 * delta_m0**3 + 2 * D14 * D51 * delta_m0 + 36 * D14 * D53 * delta_m0**3 + 720 * D14 * D55 * delta_m0**5 + 20 * D15 * D52 * delta_m0**3 + 720 * D15 * D54 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D11 * D53 * delta_m0 + (8 * D12 * D52 * delta_m0) / 3 + 16 * D12 * D54 * delta_m0**3 + 2 * D13 * D51 * delta_m0 + 24 * D13 * D53 * delta_m0**3 + 240 * D13 * D55 * delta_m0**5 + 16 * D14 * D52 * delta_m0**3 + 384 * D14 * D54 * delta_m0**5 + 240 * D15 * D53 * delta_m0**5 + 9600 * D15 * D55 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D11 * D52 * delta_m0 + 2 * D12 * D51 * delta_m0 + 12 * D12 * D53 * delta_m0**3 + 12 * D13 * D52 * delta_m0**3 + 144 * D13 * D54 * delta_m0**5 + 144 * D14 * D53 * delta_m0**5 + 2880 * D14 * D55 * delta_m0**7 + 2880 * D15 * D54 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D11 * D51 * delta_m0 + 8 * D12 * D52 * delta_m0**3 + 72 * D13 * D53 * delta_m0**5 + 1152 * D14 * D54 * delta_m0**7 + 28800 * D15 * D55 * delta_m0**9)

    coeff_C2C2 = \
        (x_p1**9 - x_m1**9) * (25 * D25**2 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D24 * D25 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D23 * D25  * delta_m0) / 7 + (16 * D24**2 * delta_m0) / 7 + (400 * D25**2 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D22 * D25  * delta_m0) / 3 + 4 * D23 * D24  * delta_m0 + 80 * D24 * D25 * delta_m0**3) + (x_p1**5 - x_m1**5) * (2 * D21 * D25  * delta_m0 + (16 * D22 * D24  * delta_m0) / 5 + (9 * D23**2 * delta_m0) / 5 + 48 * D23 * D25  * delta_m0**3 + (144 * D24**2 * delta_m0**3) / 5 + 720 * D25**2 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D21 * D24  * delta_m0 + 3 * D22 * D23  * delta_m0 + 20 * D22 * D25  * delta_m0**3 + 36 * D23 * D24  * delta_m0**3 + 720 * D24 * D25 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D21 * D23  * delta_m0 + (4 * D22**2 * delta_m0) / 3 + 16 * D22 * D24  * delta_m0**3 + 12 * D23**2 * delta_m0**3 + 240 * D23 * D25  * delta_m0**5 + 192 * D24**2 * delta_m0**5 + 4800 * D25**2 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D21 * D22  * delta_m0 + 12 * D22 * D23  * delta_m0**3 + 144 * D23 * D24  * delta_m0**5 + 2880 * D24 * D25 * delta_m0**7) \
        + (x_p1 - x_m1) * ( D21**2 * delta_m0 + 4 * D22**2 * delta_m0**3 + 36 * D23**2 * delta_m0**5 + 576 * D24**2 * delta_m0**7 + 14400 * D25**2 * delta_m0**9)
    coeff_C2C3 = \
        (x_p1**9 - x_m1**9) * (50 * D25 * D35 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D24 * D35 * delta_m0 + 5 * D25 * D34 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D23 * D35  * delta_m0) / 7 + (32 * D24 * D34 * delta_m0) / 7 + (30 * D25 * D33 * delta_m0) / 7 + (800 * D25 * D35 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D22 * D35  * delta_m0) / 3 + 4 * D23 * D34  * delta_m0 + 4 * D24 * D33 * delta_m0 + 80 * D24 * D35 * delta_m0**3 + (10 * D25 * D32 * delta_m0) / 3 + 80 * D25 * D34 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D21 * D35  * delta_m0 + (16 * D22 * D34  * delta_m0) / 5 + (18 * D23 * D33 * delta_m0) / 5 + 48 * D23 * D35  * delta_m0**3 + (16 * D24 * D32 * delta_m0) / 5 + (288 * D24 * D34 * delta_m0**3) / 5 + 2 * D25 * D31 * delta_m0 + 48 * D25 * D33 * delta_m0**3 + 1440 * D25 * D35 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D21 * D34  * delta_m0 + 3 * D22 * D33 * delta_m0 + 20 * D22 * D35  * delta_m0**3 + 3 * D23 * D32 * delta_m0 + 36 * D23 * D34  * delta_m0**3 + 2 * D24 * D31 * delta_m0 + 36 * D24 * D33 * delta_m0**3 + 720 * D24 * D35 * delta_m0**5 + 20 * D25 * D32 * delta_m0**3 + 720 * D25 * D34 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D21 * D33 * delta_m0 + (8 * D22 * D32 * delta_m0) / 3 + 16 * D22 * D34  * delta_m0**3 + 2 * D23 * D31 * delta_m0 + 24 * D23 * D33 * delta_m0**3 + 240 * D23 * D35  * delta_m0**5 + 16 * D24 * D32 * delta_m0**3 + 384 * D24 * D34 * delta_m0**5 + 240 * D25 * D33 * delta_m0**5 + 9600 * D25 * D35 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D21 * D32 * delta_m0 + 2 * D22 * D31 * delta_m0 + 12 * D22 * D33 * delta_m0**3 + 12 * D23 * D32 * delta_m0**3 + 144 * D23 * D34  * delta_m0**5 + 144 * D24 * D33 * delta_m0**5 + 2880 * D24 * D35 * delta_m0**7 + 2880 * D25 * D34 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D21 * D31 * delta_m0 + 8 * D22 * D32 * delta_m0**3 + 72 * D23 * D33 * delta_m0**5 + 1152 * D24 * D34 * delta_m0**7 + 28800 * D25 * D35 * delta_m0**9)
    coeff_C2C4 = \
        (x_p1**9 - x_m1**9) * (50 * D25 * D45 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D24 * D45 * delta_m0 + 5 * D25 * D44 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D23 * D45  * delta_m0) / 7 + (32 * D24 * D44 * delta_m0) / 7 + (30 * D25 * D43 * delta_m0) / 7 + (800 * D25 * D45 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D22 * D45  * delta_m0) / 3 + 4 * D23 * D44 * delta_m0 + 4 * D24 * D43 * delta_m0 + 80 * D24 * D45 * delta_m0**3 + (10 * D25 * D42 * delta_m0) / 3 + 80 * D25 * D44 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D21 * D45  * delta_m0 + (16 * D22 * D44 * delta_m0) / 5 + (18 * D23 * D43 * delta_m0) / 5 + 48 * D23 * D45  * delta_m0**3 + (16 * D24 * D42 * delta_m0) / 5 + (288 * D24 * D44 * delta_m0**3) / 5 + 2 * D25 * D41 * delta_m0 + 48 * D25 * D43 * delta_m0**3 + 1440 * D25 * D45 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D21 * D44 * delta_m0 + 3 * D22 * D43 * delta_m0 + 20 * D22 * D45  * delta_m0**3 + 3 * D23 * D42 * delta_m0 + 36 * D23 * D44 * delta_m0**3 + 2 * D24 * D41 * delta_m0 + 36 * D24 * D43 * delta_m0**3 + 720 * D24 * D45 * delta_m0**5 + 20 * D25 * D42 * delta_m0**3 + 720 * D25 * D44 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D21 * D43 * delta_m0 + (8 * D22 * D42 * delta_m0) / 3 + 16 * D22 * D44 * delta_m0**3 + 2 * D23 * D41 * delta_m0 + 24 * D23 * D43 * delta_m0**3 + 240 * D23 * D45  * delta_m0**5 + 16 * D24 * D42 * delta_m0**3 + 384 * D24 * D44 * delta_m0**5 + 240 * D25 * D43 * delta_m0**5 + 9600 * D25 * D45 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D21 * D42 * delta_m0 + 2 * D22 * D41 * delta_m0 + 12 * D22 * D43 * delta_m0**3 + 12 * D23 * D42 * delta_m0**3 + 144 * D23 * D44 * delta_m0**5 + 144 * D24 * D43 * delta_m0**5 + 2880 * D24 * D45 * delta_m0**7 + 2880 * D25 * D44 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D21 * D41 * delta_m0 + 8 * D22 * D42 * delta_m0**3 + 72 * D23 * D43 * delta_m0**5 + 1152 * D24 * D44 * delta_m0**7 + 28800 * D25 * D45 * delta_m0**9)
    coeff_C2C5 = \
        (x_p1**9 - x_m1**9) * (50 * D25 * D55 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D24 * D55 * delta_m0 + 5 * D25 * D54 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D23 * D55  * delta_m0) / 7 + (32 * D24 * D54 * delta_m0) / 7 + (30 * D25 * D53 * delta_m0) / 7 + (800 * D25 * D55 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D22 * D55  * delta_m0) / 3 + 4 * D23 * D54 * delta_m0 + 4 * D24 * D53 * delta_m0 + 80 * D24 * D55 * delta_m0**3 + (10 * D25 * D52 * delta_m0) / 3 + 80 * D25 * D54 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D21 * D55  * delta_m0 + (16 * D22 * D54 * delta_m0) / 5 + (18 * D23 * D53 * delta_m0) / 5 + 48 * D23 * D55  * delta_m0**3 + (16 * D24 * D52 * delta_m0) / 5 + (288 * D24 * D54 * delta_m0**3) / 5 + 2 * D25 * D51 * delta_m0 + 48 * D25 * D53 * delta_m0**3 + 1440 * D25 * D55 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D21 * D54 * delta_m0 + 3 * D22 * D53 * delta_m0 + 20 * D22 * D55  * delta_m0**3 + 3 * D23 * D52 * delta_m0 + 36 * D23 * D54 * delta_m0**3 + 2 * D24 * D51 * delta_m0 + 36 * D24 * D53 * delta_m0**3 + 720 * D24 * D55 * delta_m0**5 + 20 * D25 * D52 * delta_m0**3 + 720 * D25 * D54 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D21 * D53 * delta_m0 + (8 * D22 * D52 * delta_m0) / 3 + 16 * D22 * D54 * delta_m0**3 + 2 * D23 * D51 * delta_m0 + 24 * D23 * D53 * delta_m0**3 + 240 * D23 * D55  * delta_m0**5 + 16 * D24 * D52 * delta_m0**3 + 384 * D24 * D54 * delta_m0**5 + 240 * D25 * D53 * delta_m0**5 + 9600 * D25 * D55 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D21 * D52 * delta_m0 + 2 * D22 * D51 * delta_m0 + 12 * D22 * D53 * delta_m0**3 + 12 * D23 * D52 * delta_m0**3 + 144 * D23 * D54 * delta_m0**5 + 144 * D24 * D53 * delta_m0**5 + 2880 * D24 * D55 * delta_m0**7 + 2880 * D25 * D54 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D21 * D51 * delta_m0 + 8 * D22 * D52 * delta_m0**3 + 72 * D23 * D53 * delta_m0**5 + 1152 * D24 * D54 * delta_m0**7 + 28800 * D25 * D55 * delta_m0**9)

    coeff_C3C3 = \
        (x_p1**9 - x_m1**9) * (25 * D35**2 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D34 * D35 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D33 * D35 * delta_m0) / 7 + (16 * D34**2 * delta_m0) / 7 + (400 * D35**2 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D32 * D35 * delta_m0) / 3 + 4 * D33 * D34 * delta_m0 + 80 * D34 * D35 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D31 * D35 * delta_m0 + (16 * D32 * D34 * delta_m0) / 5 + (9 * D33**2 * delta_m0) / 5 + 48 * D33 * D35 * delta_m0**3 + (144 * D34**2 * delta_m0**3) / 5 + 720 * D35**2 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D31 * D34 * delta_m0 + 3 * D32 * D33 * delta_m0 + 20 * D32 * D35 * delta_m0**3 + 36 * D33 * D34 * delta_m0**3 + 720 * D34 * D35 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D31 * D33 * delta_m0 + (4 * D32**2 * delta_m0) / 3 + 16 * D32 * D34 * delta_m0**3 + 12 * D33**2 * delta_m0**3 + 240 * D33 * D35 * delta_m0**5 + 192 * D34**2 * delta_m0**5 + 4800 * D35**2 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D31 * D32 * delta_m0 + 12 * D32 * D33 * delta_m0**3 + 144 * D33 * D34 * delta_m0**5 + 2880 * D34 * D35 * delta_m0**7) \
        + (x_p1 - x_m1) * ( D31**2 * delta_m0 + 4 * D32**2 * delta_m0**3 + 36 * D33**2 * delta_m0**5 + 576 * D34**2 * delta_m0**7 + 14400 * D35**2 * delta_m0**9)
    coeff_C3C4 = \
        (x_p1**9 - x_m1**9) * (50 * D35 * D45 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D34 * D45 * delta_m0 + 5 * D35 * D44 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D33 * D45 * delta_m0) / 7 + (32 * D34 * D44 * delta_m0) / 7 + (30 * D35 * D43 * delta_m0) / 7 + (800 * D35 * D45 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D32 * D45 * delta_m0) / 3 + 4 * D33 * D44 * delta_m0 + 4 * D34 * D43 * delta_m0 + 80 * D34 * D45 * delta_m0**3 + (10 * D35 * D42 * delta_m0) / 3 + 80 * D35 * D44 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D31 * D45 * delta_m0 + (16 * D32 * D44 * delta_m0) / 5 + (18 * D33 * D43 * delta_m0) / 5 + 48 * D33 * D45 * delta_m0**3 + (16 * D34 * D42 * delta_m0) / 5 + (288 * D34 * D44 * delta_m0**3) / 5 + 2 * D35 * D41 * delta_m0 + 48 * D35 * D43 * delta_m0**3 + 1440 * D35 * D45 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D31 * D44 * delta_m0 + 3 * D32 * D43 * delta_m0 + 20 * D32 * D45 * delta_m0**3 + 3 * D33 * D42 * delta_m0 + 36 * D33 * D44 * delta_m0**3 + 2 * D34 * D41 * delta_m0 + 36 * D34 * D43 * delta_m0**3 + 720 * D34 * D45 * delta_m0**5 + 20 * D35 * D42 * delta_m0**3 + 720 * D35 * D44 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D31 * D43 * delta_m0 + (8 * D32 * D42 * delta_m0) / 3 + 16 * D32 * D44 * delta_m0**3 + 2 * D33 * D41 * delta_m0 + 24 * D33 * D43 * delta_m0**3 + 240 * D33 * D45 * delta_m0**5 + 16 * D34 * D42 * delta_m0**3 + 384 * D34 * D44 * delta_m0**5 + 240 * D35 * D43 * delta_m0**5 + 9600 * D35 * D45 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D31 * D42 * delta_m0 + 2 * D32 * D41 * delta_m0 + 12 * D32 * D43 * delta_m0**3 + 12 * D33 * D42 * delta_m0**3 + 144 * D33 * D44 * delta_m0**5 + 144 * D34 * D43 * delta_m0**5 + 2880 * D34 * D45 * delta_m0**7 + 2880 * D35 * D44 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D31 * D41 * delta_m0 + 8 * D32 * D42 * delta_m0**3 + 72 * D33 * D43 * delta_m0**5 + 1152 * D34 * D44 * delta_m0**7 + 28800 * D35 * D45 * delta_m0**9)
    coeff_C3C5 = \
        (x_p1**9 - x_m1**9) * (50 * D35 * D55 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D34 * D55 * delta_m0 + 5 * D35 * D54 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D33 * D55 * delta_m0) / 7 + (32 * D34 * D54 * delta_m0) / 7 + (30 * D35 * D53 * delta_m0) / 7 + (800 * D35 * D55 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D32 * D55 * delta_m0) / 3 + 4 * D33 * D54 * delta_m0 + 4 * D34 * D53 * delta_m0 + 80 * D34 * D55 * delta_m0**3 + (10 * D35 * D52 * delta_m0) / 3 + 80 * D35 * D54 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D31 * D55 * delta_m0 + (16 * D32 * D54 * delta_m0) / 5 + (18 * D33 * D53 * delta_m0) / 5 + 48 * D33 * D55 * delta_m0**3 + (16 * D34 * D52 * delta_m0) / 5 + (288 * D34 * D54 * delta_m0**3) / 5 + 2 * D35 * D51 * delta_m0 + 48 * D35 * D53 * delta_m0**3 + 1440 * D35 * D55 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D31 * D54 * delta_m0 + 3 * D32 * D53 * delta_m0 + 20 * D32 * D55 * delta_m0**3 + 3 * D33 * D52 * delta_m0 + 36 * D33 * D54 * delta_m0**3 + 2 * D34 * D51 * delta_m0 + 36 * D34 * D53 * delta_m0**3 + 720 * D34 * D55 * delta_m0**5 + 20 * D35 * D52 * delta_m0**3 + 720 * D35 * D54 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D31 * D53 * delta_m0 + (8 * D32 * D52 * delta_m0) / 3 + 16 * D32 * D54 * delta_m0**3 + 2 * D33 * D51 * delta_m0 + 24 * D33 * D53 * delta_m0**3 + 240 * D33 * D55 * delta_m0**5 + 16 * D34 * D52 * delta_m0**3 + 384 * D34 * D54 * delta_m0**5 + 240 * D35 * D53 * delta_m0**5 + 9600 * D35 * D55 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D31 * D52 * delta_m0 + 2 * D32 * D51 * delta_m0 + 12 * D32 * D53 * delta_m0**3 + 12 * D33 * D52 * delta_m0**3 + 144 * D33 * D54 * delta_m0**5 + 144 * D34 * D53 * delta_m0**5 + 2880 * D34 * D55 * delta_m0**7 + 2880 * D35 * D54 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D31 * D51 * delta_m0 + 8 * D32 * D52 * delta_m0**3 + 72 * D33 * D53 * delta_m0**5 + 1152 * D34 * D54 * delta_m0**7 + 28800 * D35 * D55 * delta_m0**9)

    coeff_C4C4 = \
        (x_p1**9 - x_m1**9) * (25 * D45**2 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D44 * D45 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D43 * D45 * delta_m0) / 7 + (16 * D44**2 * delta_m0) / 7 + (400 * D45**2 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D42 * D45 * delta_m0) / 3 + 4 * D43 * D44 * delta_m0 + 80 * D44 * D45 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D41 * D45 * delta_m0 + (16 * D42 * D44 * delta_m0) / 5 + (9 * D43**2 * delta_m0) / 5 + 48 * D43 * D45 * delta_m0**3 + (144 * D44**2 * delta_m0**3) / 5 + 720 * D45**2 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D41 * D44 * delta_m0 + 3 * D42 * D43 * delta_m0 + 20 * D42 * D45 * delta_m0**3 + 36 * D43 * D44 * delta_m0**3 + 720 * D44 * D45 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D41 * D43 * delta_m0 + (4 * D42**2 * delta_m0) / 3 + 16 * D42 * D44 * delta_m0**3 + 12 * D43**2 * delta_m0**3 + 240 * D43 * D45 * delta_m0**5 + 192 * D44**2 * delta_m0**5 + 4800 * D45**2 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D41 * D42 * delta_m0 + 12 * D42 * D43 * delta_m0**3 + 144 * D43 * D44 * delta_m0**5 + 2880 * D44 * D45 * delta_m0**7) \
        + (x_p1 - x_m1) * ( D41**2 * delta_m0 + 4 * D42**2 * delta_m0**3 + 36 * D43**2 * delta_m0**5 + 576 * D44**2 * delta_m0**7 + 14400 * D45**2 * delta_m0**9)
    coeff_C4C5 = \
        (x_p1**9 - x_m1**9) * (50 * D45 * D55 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D44 * D55 * delta_m0 + 5 * D45 * D54 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D43 * D55 * delta_m0) / 7 + (32 * D44 * D54 * delta_m0) / 7 + (30 * D45 * D53 * delta_m0) / 7 + (800 * D45 * D55 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D42 * D55 * delta_m0) / 3 + 4 * D43 * D54 * delta_m0 + 4 * D44 * D53 * delta_m0 + 80 * D44 * D55 * delta_m0**3 + (10 * D45 * D52 * delta_m0) / 3 + 80 * D45 * D54 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D41 * D55 * delta_m0 + (16 * D42 * D54 * delta_m0) / 5 + (18 * D43 * D53 * delta_m0) / 5 + 48 * D43 * D55 * delta_m0**3 + (16 * D44 * D52 * delta_m0) / 5 + (288 * D44 * D54 * delta_m0**3) / 5 + 2 * D45 * D51 * delta_m0 + 48 * D45 * D53 * delta_m0**3 + 1440 * D45 * D55 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D41 * D54 * delta_m0 + 3 * D42 * D53 * delta_m0 + 20 * D42 * D55 * delta_m0**3 + 3 * D43 * D52 * delta_m0 + 36 * D43 * D54 * delta_m0**3 + 2 * D44 * D51 * delta_m0 + 36 * D44 * D53 * delta_m0**3 + 720 * D44 * D55 * delta_m0**5 + 20 * D45 * D52 * delta_m0**3 + 720 * D45 * D54 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D41 * D53 * delta_m0 + (8 * D42 * D52 * delta_m0) / 3 + 16 * D42 * D54 * delta_m0**3 + 2 * D43 * D51 * delta_m0 + 24 * D43 * D53 * delta_m0**3 + 240 * D43 * D55 * delta_m0**5 + 16 * D44 * D52 * delta_m0**3 + 384 * D44 * D54 * delta_m0**5 + 240 * D45 * D53 * delta_m0**5 + 9600 * D45 * D55 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D41 * D52 * delta_m0 + 2 * D42 * D51 * delta_m0 + 12 * D42 * D53 * delta_m0**3 + 12 * D43 * D52 * delta_m0**3 + 144 * D43 * D54 * delta_m0**5 + 144 * D44 * D53 * delta_m0**5 + 2880 * D44 * D55 * delta_m0**7 + 2880 * D45 * D54 * delta_m0**7) \
        + (x_p1 - x_m1) * (2 * D41 * D51 * delta_m0 + 8 * D42 * D52 * delta_m0**3 + 72 * D43 * D53 * delta_m0**5 + 1152 * D44 * D54 * delta_m0**7 + 28800 * D45 * D55 * delta_m0**9)

    coeff_C5C5 = \
        (x_p1**9 - x_m1**9) * (25 * D55**2 * delta_m0) / 9 \
        + (x_p1**8 - x_m1**8) * (5 * D54 * D55 * delta_m0) \
        + (x_p1**7 - x_m1**7) * ((30 * D53 * D55 * delta_m0) / 7 + (16 * D54**2 * delta_m0) / 7 + (400 * D55**2 * delta_m0**3) / 7) \
        + (x_p1**6 - x_m1**6) * ((10 * D52 * D55 * delta_m0) / 3 + 4 * D53 * D54 * delta_m0 + 80 * D54 * D55 * delta_m0**3) \
        + (x_p1**5 - x_m1**5) * (2 * D51 * D55 * delta_m0 + (16 * D52 * D54 * delta_m0) / 5 + (9 * D53**2 * delta_m0) / 5 + 48 * D53 * D55 * delta_m0**3 + (144 * D54**2 * delta_m0**3) / 5 + 720 * D55**2 * delta_m0**5) \
        + (x_p1**4 - x_m1**4) * (2 * D51 * D54 * delta_m0 + 3 * D52 * D53 * delta_m0 + 20 * D52 * D55 * delta_m0**3 + 36 * D53 * D54 * delta_m0**3 + 720 * D54 * D55 * delta_m0**5) \
        + (x_p1**3 - x_m1**3) * (2 * D51 * D53 * delta_m0 + (4 * D52**2 * delta_m0) / 3 + 16 * D52 * D54 * delta_m0**3 + 12 * D53**2 * delta_m0**3 + 240 * D53 * D55 * delta_m0**5 + 192 * D54**2 * delta_m0**5 + 4800 * D55**2 * delta_m0**7) \
        + (x_p1**2 - x_m1**2) * (2 * D51 * D52 * delta_m0 + 12 * D52 * D53 * delta_m0**3 + 144 * D53 * D54 * delta_m0**5 + 2880 * D54 * D55 * delta_m0**7) \
        + (x_p1 - x_m1) * ( D51**2 * delta_m0 + 4 * D52**2 * delta_m0**3 + 36 * D53**2 * delta_m0**5 + 576 * D54**2 * delta_m0**7 + 14400 * D55**2 * delta_m0**9)

    return (C00, C10, C11, C20, C21, C22, C30, C31, C32, C33, C40, C41, C42, C43, C44,
            C50, C51, C52, C53, C54, C55), \
            (coeff_C0C0, coeff_C0C1, coeff_C0C2, coeff_C0C3, coeff_C0C4, coeff_C0C5,
             coeff_C1C1, coeff_C1C2, coeff_C1C3, coeff_C1C4, coeff_C1C5,
             coeff_C2C2, coeff_C2C3, coeff_C2C4, coeff_C2C5,
             coeff_C3C3, coeff_C3C4, coeff_C3C5,
             coeff_C4C4, coeff_C4C5,
             coeff_C5C5)
