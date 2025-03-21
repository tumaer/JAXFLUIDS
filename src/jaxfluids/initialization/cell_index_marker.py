from typing import Dict, Tuple, Callable

from jax import Array
import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.data_types.buffers import LevelsetFieldBuffers, LevelsetSolidCellIndices, LevelsetSolidCellIndicesField
from jaxfluids.data_types.numerical_setup.levelset import LevelsetSetup
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.levelset.geometry.geometry_calculator import GeometryCalculator
from jaxfluids.levelset.geometry.mask_functions import (compute_narrowband_mask, compute_fluid_masks,
                                                    compute_cut_cell_mask_sign_change_based,
                                                    compute_cut_cell_mask_value_based)
from jaxfluids.levelset.mixing.helper_functions import (
    move_source_to_target_ii, move_source_to_target_ij,
    move_source_to_target_ijk)

def compute_solid_cell_indices(
        levelset_fields: LevelsetFieldBuffers,
        geometry_calculator: GeometryCalculator,
        domain_information: DomainInformation,
        levelset_setup: LevelsetSetup
        ) -> LevelsetSolidCellIndices:
    """Computes interface, extension, and mixing cell indices. They are used
    for static fluid solid simulations to increase computational
    efficiency.

    :param levelset_fields: _description_
    :type levelset_fields: LevelsetFieldBuffers
    :return: _description_
    :rtype: LevelsetSolidCellIndices
    """
    
    active_axes_indices = domain_information.active_axes_indices
    is_parallel = domain_information.is_parallel
    host_count = domain_information.host_count
    no_subdomains = domain_information.no_subdomains
    local_subdomain_count = no_subdomains//host_count

    dim = domain_information.dim

    is_cell_based_mixing_fluid = levelset_setup.mixing.conservatives.is_cell_based_computation
    is_cell_based_mixing_solid = levelset_setup.mixing.solids.is_cell_based_computation

    is_extension_fluid = levelset_setup.extension.primitives.interpolation.is_cell_based_computation
    is_extension_solid = levelset_setup.extension.solids.interpolation.is_cell_based_computation
    is_interface_flux = levelset_setup.interface_flux.is_cell_based_computation

    solid_coupling = levelset_setup.solid_coupling

    tag_functions_dict: Dict[str, Callable] = {}
    
    if is_interface_flux:
        tag_functions_dict["interface"] = tag_interface_cells
    
    if is_extension_fluid:
        tag_functions_dict["extension_fluid"] = tag_extension_cells

    if is_extension_solid and solid_coupling.thermal == "TWO-WAY":
        raise NotImplementedError

    if is_cell_based_mixing_fluid:
        tag_functions_dict["mixing_source_fluid"] = tag_mixing_source_cells

    if is_cell_based_mixing_solid and solid_coupling.thermal == "TWO-WAY":
        raise NotImplementedError

    cell_indices_dict: Dict[str, LevelsetSolidCellIndicesField] = {}

    # NOTE looping fields, computing cell indices. For parallel runs, also computes mask
    for field, tag_function in tag_functions_dict.items():
        is_fluid = True if "fluid" in field else False
        if is_parallel:
            mask_indices, count_max, count = jax.pmap(
                tag_function, static_broadcasted_argnums=(1,2,3,4), in_axes=(0,None,None,None,None),
                out_axes=(0,None,0), axis_name="i")(levelset_fields, geometry_calculator,
                                                    domain_information, levelset_setup,
                                                    is_fluid)
            indices = jax.pmap(get_cell_indices, static_broadcasted_argnums=(1),
                               out_axes=(0), axis_name="i")(mask_indices, int(count_max))
            mask = np.zeros((local_subdomain_count,count_max),dtype=int) # TODO aaron bottleneck, nicer way inside pmap ?
            for i in range(local_subdomain_count):
                mask[i,:count[i]] = 1
        else:
            mask_indices, count_max, count = tag_function(
                levelset_fields, geometry_calculator,
                domain_information, levelset_setup,
                is_fluid)
            indices = get_cell_indices(mask_indices, count)
            mask = None

        cell_indices_dict[field] = LevelsetSolidCellIndicesField(
            indices, mask)

    mixing_targets_fluid = levelset_setup.mixing.conservatives.mixing_targets
    mixing_targets_solid = levelset_setup.mixing.solids.mixing_targets

    loop_tuple = [
        (is_cell_based_mixing_fluid, mixing_targets_fluid, "fluid"),
    ]
    if solid_coupling.thermal == "TWO-WAY":
        raise NotImplementedError

    for is_cell_based, mixing_targets, name in loop_tuple:

        is_fluid = True if name == "fluid" else False

        if is_cell_based:

            tag_functions_dict: Dict[str, Callable] = {}

            for i in active_axes_indices:
                target_str = f"ii_{i}"
                tag_functions_dict[f"mixing_target_{target_str:s}_{name:s}"] = get_tag_mixing_target_cells_function(target_str)
            if dim > 1 and mixing_targets > 1:
                for (i,j) in ((0,1),(0,2),(1,2)):
                    target_str = f"ij_{i:d}{j:d}"
                    if i in active_axes_indices and j in active_axes_indices:
                        tag_functions_dict[f"mixing_target_{target_str:s}_{name:s}"] = get_tag_mixing_target_cells_function(target_str)
            if dim == 3 and mixing_targets == 3:
                tag_functions_dict[f"mixing_target_ijk_{name:s}"] = get_tag_mixing_target_cells_function("ijk")

            source_indices = cell_indices_dict[f"mixing_source_{name:s}"].indices
            source_mask = cell_indices_dict[f"mixing_source_{name:s}"].mask
            for field, tag_function in tag_functions_dict.items():
                if is_parallel:
                    target_indices = jax.pmap(
                        tag_function, in_axes=(0,0,None,None,None,None),
                        static_broadcasted_argnums=(2,3,4,5))(source_indices, levelset_fields, geometry_calculator,
                                                            domain_information, levelset_setup, is_fluid)
                    
                else:
                    target_indices = tag_function(
                        source_indices, levelset_fields, geometry_calculator,
                        domain_information, levelset_setup, is_fluid)

                cell_indices_dict[field] = LevelsetSolidCellIndicesField(
                    target_indices, source_mask)

    solid_cell_indices = LevelsetSolidCellIndices(
        **cell_indices_dict)
    
    return solid_cell_indices

def get_cell_indices(
        mask: Array,
        count: int,
        ) -> Array:
    indices = jnp.where(mask, size=count)
    return indices

def tag_interface_cells(
    levelset_fields: LevelsetFieldBuffers,
    geometry_calculator: GeometryCalculator,
    domain_information: DomainInformation,
    levelset_setup: LevelsetSetup,
    is_fluid: bool
    ) -> Tuple[Array, Array, Array]:
    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    is_parallel = domain_information.is_parallel
    apertures = levelset_fields.apertures
    interface_length = geometry_calculator.compute_interface_length(apertures)
    mask_interface_cells = interface_length[...,nhx_,nhy_,nhz_] > 0.0
    interface_cells_count = jnp.sum(mask_interface_cells, dtype=int)
    if is_parallel:
        interface_cells_count_max = jax.lax.pmax(interface_cells_count, axis_name="i")
    else:
        interface_cells_count_max = interface_cells_count
    return mask_interface_cells, interface_cells_count_max, interface_cells_count
        
def tag_extension_cells(
    levelset_fields: LevelsetFieldBuffers,
    geometry_calculator: GeometryCalculator,
    domain_information: DomainInformation,
    levelset_setup: LevelsetSetup,
    is_fluid: bool
    ) -> Tuple[Array, Array, Array]:

    nhx_,nhy_,nhz_ = domain_information.domain_slices_geometry
    nhx,nhy,nhz = domain_information.domain_slices_conservatives
    is_parallel = domain_information.is_parallel
    cell_size = domain_information.smallest_cell_size

    levelset_model = levelset_setup.model

    levelset = levelset_fields.levelset
    volume_fraction = levelset_fields.volume_fraction
    
    narrowband_computation = levelset_setup.narrowband.computation_width
    mask_narrowband = compute_narrowband_mask(levelset[...,nhx,nhy,nhz], cell_size, narrowband_computation)
    if is_fluid:
        mask_real = compute_fluid_masks(volume_fraction[...,nhx_,nhy_,nhz_], levelset_model)
        mask_ghost = 1 - mask_real
    else:
        volume_fraction_solid = 1.0 - volume_fraction[...,nhx_,nhy_,nhz_]
        mask_real = volume_fraction_solid > 0.0
        mask_ghost = 1 - mask_real

    mask_extension_cells = mask_narrowband * mask_ghost

    extension_cells_count = jnp.sum(mask_extension_cells, dtype=int)
    if is_parallel:
        extension_cells_count_max = jax.lax.pmax(extension_cells_count, axis_name="i")
    else:
        extension_cells_count_max = extension_cells_count
    
    return mask_extension_cells, extension_cells_count_max, extension_cells_count

def tag_mixing_source_cells(
        levelset_fields: LevelsetFieldBuffers,
        geometry_calculator: GeometryCalculator,
        domain_information: DomainInformation,
        levelset_setup: LevelsetSetup,
        is_fluid: bool
        ) -> Tuple[Array, Array, Array]:

    nh_ = domain_information.nh_geometry
    nhx__,nhy__,nhz__ = domain_information.domain_slices_conservatives_to_geometry
    is_parallel = domain_information.is_parallel
    cell_size = domain_information.smallest_cell_size
    nh_offset = domain_information.nh_offset

    mixing_setup = levelset_setup.mixing

    levelset = levelset_fields.levelset
    apertures = levelset_fields.apertures
    volume_fraction = levelset_fields.volume_fraction

    # NOTE offset of 1 because mixing stencil requires 1 halo cell
    active_axes_indices = domain_information.active_axes_indices
    s_ = (...,)
    for i in range(3):
        if i in active_axes_indices:
            s_ += (jnp.s_[nh_-1:-nh_+1],)
        else:
            s_ += (jnp.s_[:],)

    cut_cell_mask = compute_cut_cell_mask_sign_change_based(levelset, nh_offset)
    
    if is_fluid:
        volume_fraction_threshold = mixing_setup.conservatives.volume_fraction_threshold
    else:
        volume_fraction_threshold = mixing_setup.solids.volume_fraction_threshold
        volume_fraction = 1.0 - volume_fraction

    source_mask = volume_fraction < volume_fraction_threshold
    source_mask *= cut_cell_mask
    source_mask = source_mask[s_]
    
    count = jnp.sum(source_mask, dtype=int)
    if is_parallel:
        count_max = jax.lax.pmax(count, axis_name="i")
    else:
        count_max = count

    return source_mask, count_max, count

def get_tag_mixing_target_cells_function(
    target: str
    ) -> Callable:

    def wrapper(
        source_indices: Tuple[Array],
        levelset_fields: LevelsetFieldBuffers,
        geometry_calculator: GeometryCalculator,
        domain_information: DomainInformation,
        levelset_setup: LevelsetSetup,
        is_fluid: bool
        ) -> Tuple[Array, Array, Array]:


        nh_ = domain_information.nh_geometry
        active_axes_indices = domain_information.active_axes_indices

        normal_computation_method = levelset_setup.mixing.conservatives.normal_computation_method

        levelset = levelset_fields.levelset
        apertures = levelset_fields.apertures

        if normal_computation_method == "FINITEDIFFERENCE":
            normal = geometry_calculator.compute_normal(levelset)
        elif normal_computation_method == "APERTUREBASED":
            normal = geometry_calculator.compute_normal_apertures_based(apertures)
        else:
            raise NotImplementedError
        
        if not is_fluid:
            normal = -normal

        # NOTE offset of 1 because mixing stencil requires 1 halo cell
        s_ = (...,)
        for i in range(3):
            if i in active_axes_indices:
                s_ += (jnp.s_[nh_-1:-nh_+1],)
            else:
                s_ += (jnp.s_[:],)
                
        normal_sign = jnp.sign(normal).astype(int)
        normal_sign = normal_sign[s_]
        normal_sign = normal_sign[(...,)+ source_indices]

        target_indices = ()

        if "ii_" in target:
            for axis_index in range(3):
                if target == "ii_%d" % axis_index:

                    target_indices = ()
                    for i in range(3):
                        if i == axis_index:
                            src_ = source_indices[i] + normal_sign[i]
                        else:
                            src_ = source_indices[i]
                        target_indices += (src_,)


        elif "ij_" in target:
            for (axis_i, axis_j) in ((0,1),(0,2),(1,2)):
                if target == "ij_%d%d" % (axis_i, axis_j):

                    target_indices = ()
                    for i in range(3):
                        if i == axis_i or i == axis_j:
                            src_ = source_indices[i] + normal_sign[i]
                        else:
                            src_ = source_indices[i]
                        target_indices += (src_,)

        elif "ijk" in target:
            target_indices = ()
            for i in range(3):
                src_ = source_indices[i] + normal_sign[i]
                target_indices += (src_,)

        return target_indices


    return wrapper
    