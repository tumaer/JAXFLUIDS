from typing import Dict, Tuple

import jax.numpy as jnp
import jax

from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.levelset.extension.iterative_extender import IterativeExtender
from jaxfluids.data_types.numerical_setup.levelset import NarrowBandSetup, IterativeExtensionSetup
from jaxfluids.data_types.information import LevelsetProcedureInformation
from jaxfluids.levelset.geometry.mask_functions import (
    compute_cut_cell_mask_sign_change_based,
    compute_cut_cell_mask_vf_based,
    compute_cut_cell_mask_value_based,
    compute_narrowband_mask,)

Array = jax.Array


def solve_interface_interaction(
        primitives: Array,
        normal: Array,
        curvature: Array,
        material_manager: MaterialManager,
        domain_information: DomainInformation
        ) -> Tuple[Array, Array]:
    """Solves the two-material Riemann problem for FLUID-FLUID interface interactions.

    :param primitives: Primitive variable buffer
    :type primitives: Array
    :param normal: Interface normal buffer
    :type normal: Array
    :param curvature: Interface curvature buffer
    :type curvature: Array
    :return: Interface velocity and interface pressure
    :rtype: Tuple[Array, Array]
    """

    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nhx__,nhy__,nhz__ = domain_information.domain_slices_conservatives_to_geometry

    pressure = primitives[4,...,nhx__,nhy__,nhz__]
    density = primitives[0,...,nhx__,nhy__,nhz__]
    velocity = primitives[1:4,...,nhx__,nhy__,nhz__]

    velocity_normal_projection = jnp.einsum('ijklm, ijklm -> jklm', velocity, jnp.expand_dims(normal, axis=1) ) # TODO use aperture based normal here?
    speed_of_sound = material_manager.get_speed_of_sound(pressure=pressure, density=density)
    impendance = speed_of_sound * density
    inverse_impendace_sum = 1.0 / ( impendance[0] + impendance[1] + 1e-30 )

    if curvature is not None:
        delta_p = material_manager.get_sigma() * curvature
    else:
        delta_p = 0.0

    interface_velocity = ( impendance[1] * velocity_normal_projection[1] + impendance[0] * velocity_normal_projection[0] + \
                            pressure[1] - pressure[0] - delta_p ) * inverse_impendace_sum
    interface_pressure_positive = (impendance[1] * pressure[0] + impendance[0] * (pressure[1] - delta_p) + \
                                    impendance[0] * impendance[1] * (velocity_normal_projection[1] - velocity_normal_projection[0]) ) * inverse_impendace_sum
    interface_pressure_negative = (impendance[1] * (pressure[0] + delta_p) + impendance[0] * pressure[1] + \
                                    impendance[0] * impendance[1] * (velocity_normal_projection[1] - velocity_normal_projection[0]) ) * inverse_impendace_sum

    interface_pressure = jnp.stack([interface_pressure_positive, interface_pressure_negative], axis=0)

    return interface_velocity, interface_pressure


# NOTE free functions as we need this in init manager and sim manager
def compute_interface_quantities(
        primitives: Array,
        levelset: Array,
        volume_fraction: Array,
        normal: Array,
        curvature: Array,
        material_manager: MaterialManager,
        extender: IterativeExtender,
        iterative_setup: IterativeExtensionSetup,
        narrowband_setup: NarrowBandSetup,
        interface_velocity_old: Array = None,
        interface_pressure_old: Array = None,
        is_initialization: bool = False,
        ml_setup: MachineLearningSetup = None
        ) -> Tuple[Array, Array, LevelsetProcedureInformation]:
    """Computes interface velocity and pressure for
    FLUID-FLUID interface interaction and
    extends the values into the narrowband_computation.
    If interface_velocity/pressure_old buffer is
    provided, they will be used as starting values
    in the extension procedure, otherwise 0.0 will
    be used as starting values. If steps is provided,
    then steps are forced, neglecting the
    residual threshold of extender.

    :param primitives: _description_
    :type primitives: Array
    :param levelset: _description_
    :type levelset: Array
    :param volume_fraction: _description_
    :type volume_fraction: Array
    :param normal: _description_
    :type normal: Array
    :param curvature: _description_
    :type curvature: Array
    :param material_manager: _description_
    :type material_manager: MaterialManager
    :param extender: _description_
    :type extender: IterativeExtender
    :param interface_velocity_old: _description_, defaults to None
    :type interface_velocity_old: Array, optional
    :param interface_pressure_old: _description_, defaults to None
    :type interface_pressure_old: Array, optional
    :param steps: _description_, defaults to None
    :type steps: int, optional
    :param CFL: _description_, defaults to None
    :type CFL: float, optional
    :return: _description_
    :rtype: Tuple[Array, Array, LevelsetProcedureInformation]
    """

    domain_information = extender.domain_information
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nhx__,nhy__,nhz__ = domain_information.domain_slices_conservatives_to_geometry
    cell_size = domain_information.smallest_cell_size
    nh_offset = domain_information.nh_offset

    narrowband_computation = narrowband_setup.computation_width

    interface_velocity, interface_pressure = solve_interface_interaction(
        primitives, normal, curvature, material_manager, domain_information)

    mask_narrowband = compute_narrowband_mask(levelset, cell_size, narrowband_computation)
    mask_narrowband = mask_narrowband[nhx,nhy,nhz]
    # interface on cell face must be considered
    cut_cell_mask = compute_cut_cell_mask_value_based(levelset[nhx__,nhy__,nhz__], cell_size)
    inverse_cut_cell_mask = 1 - cut_cell_mask
    mask_extend = inverse_cut_cell_mask[nhx_,nhy_,nhz_] * mask_narrowband
    normal_extend = normal * jnp.sign(levelset[nhx__,nhy__,nhz__])

    interface_velocity *= cut_cell_mask
    interface_pressure *= cut_cell_mask

    if interface_velocity_old is not None:
        interface_velocity = interface_velocity + interface_velocity_old * (1-cut_cell_mask)
    if interface_pressure_old is not None:
        interface_pressure = interface_pressure + interface_pressure_old * (1-cut_cell_mask)

    interface_quantities = jnp.concatenate([
        jnp.expand_dims(interface_velocity, axis=0),
        interface_pressure], axis=0)
    
    if is_initialization:
        CFL = 0.5
        steps = 50
    else:
        CFL = iterative_setup.CFL
        steps = iterative_setup.steps

    interface_quantities, info = extender.extend(
        interface_quantities, normal_extend, mask_extend,
        0.0, CFL, steps, ml_setup=ml_setup)

    interface_quantities = interface_quantities.at[...,nhx_,nhy_,nhz_].mul(mask_narrowband)

    interface_velocity = interface_quantities[0]
    interface_pressure = interface_quantities[1:]

    return interface_velocity, interface_pressure, info



