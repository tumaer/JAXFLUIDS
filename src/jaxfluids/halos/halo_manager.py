
from __future__ import annotations
from typing import Tuple, Dict, TYPE_CHECKING

import jax
import jax.numpy as jnp, numpy as np

from jaxfluids.data_types.ml_buffers import MachineLearningSetup
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.inner.material import HaloCommunicationMaterial
from jaxfluids.halos.inner.levelset import HaloCommunicationLevelset
from jaxfluids.halos.inner.mesh import HaloCommunicationMesh
from jaxfluids.halos.inner.solids import HaloCommunicationSolids
from jaxfluids.halos.outer.levelset import BoundaryConditionLevelset
from jaxfluids.halos.outer.material import BoundaryConditionMaterial
from jaxfluids.halos.outer.conservative_mixing import BoundaryConditionConservativeMixing
from jaxfluids.halos.outer.diffuse_curvature import BoundaryConditionDiffuseCurvature
from jaxfluids.halos.outer.solids_mixing import BoundaryConditionSolidsMixing
from jaxfluids.halos.outer.fluxes import BoundaryConditionFlux
from jaxfluids.halos.outer.mesh import BoundaryConditionMesh
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.case_setup import BoundaryConditionSetup
from jaxfluids.halos.outer.solids import BoundaryConditionSolids
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup import NumericalSetup
    from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager


Array = jax.Array

class HaloManager:
    """ The HaloManager handles the halo cells,
    i.e. fills the halo cells according to the
    outer and inner (for parallelization) boundary
    conditions.
    """
    def __init__(
            self,
            numerical_setup: NumericalSetup,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            boundary_conditions_setup: BoundaryConditionSetup,
            solid_properties_manager: SolidPropertiesManager
            ) -> None:

        self.domain_information = domain_information
        self.is_parallel = domain_information.is_parallel
        self.dim = self.domain_information.dim
        self.mixing_targets_conservatives = numerical_setup.levelset.mixing.conservatives.mixing_targets
        self.mixing_targets_solids = numerical_setup.levelset.mixing.solids.mixing_targets
        
        levelset_model = numerical_setup.levelset.model
        diffuse_interface_model = numerical_setup.diffuse_interface.model

        boundary_conditions_material = boundary_conditions_setup.primitives
        boundary_conditions_levelset = boundary_conditions_setup.levelset
        boundary_conditions_solids = boundary_conditions_setup.solids

        # OUTER BOUNDARY CONDITIONS
        self.boundary_condition_material = BoundaryConditionMaterial(    
            domain_information=domain_information,
            equation_manager=equation_manager,
            boundary_conditions=boundary_conditions_material)
        if self.is_parallel:
            self.halo_communication_material = HaloCommunicationMaterial(
                domain_information = domain_information,
                equation_manager = equation_manager,
                boundary_conditions = boundary_conditions_material)

        self.boundary_condition_mesh = BoundaryConditionMesh(
            domain_information = domain_information,
            boundary_conditions = boundary_conditions_material)
        if self.is_parallel:
            self.halo_communication_mesh = HaloCommunicationMesh(
                domain_information = domain_information,
                boundary_conditions = boundary_conditions_material)

        if levelset_model:
            self.boundary_condition_levelset = BoundaryConditionLevelset(
                domain_information = domain_information,
                boundary_conditions = boundary_conditions_levelset)
            self.boundary_condition_conservative_mixing = BoundaryConditionConservativeMixing(
                domain_information = domain_information,
                boundary_conditions = boundary_conditions_material,
                equation_information = equation_manager.equation_information)
            if self.is_parallel:
                self.halo_communication_levelset = HaloCommunicationLevelset(
                    domain_information = domain_information,
                    boundary_conditions = boundary_conditions_levelset)

        solid_coupling = numerical_setup.levelset.solid_coupling
        if solid_coupling.thermal == "TWO-WAY":
            self.boundary_condition_solids = BoundaryConditionSolids(
                domain_information = domain_information,
                boundary_conditions = boundary_conditions_solids,
                solid_properties_manager = solid_properties_manager)
            self.boundary_condition_solids_mixing = BoundaryConditionSolidsMixing(
                domain_information = domain_information,
                boundary_conditions = boundary_conditions_solids
                )
            if self.is_parallel:
                self.halo_communication_solids = HaloCommunicationSolids(
                    domain_information = domain_information,
                    solid_properties_manager = solid_properties_manager,  
                    boundary_conditions = boundary_conditions_levelset
                    )

        if diffuse_interface_model:
            self.boundary_condition_curvature = BoundaryConditionDiffuseCurvature(
                domain_information = domain_information,
                boundary_conditions = boundary_conditions_material)
            
            self.boundary_condition_flux = BoundaryConditionFlux(
                domain_information=domain_information,
                boundary_conditions=boundary_conditions_material)

        equation_information = equation_manager.equation_information
        is_solid_levelset = equation_information.is_solid_levelset
        active_physics = equation_information.active_physics
        equation_type = equation_information.equation_type

        is_viscous_flux = active_physics.is_viscous_flux
        is_heat_flux = active_physics.is_heat_flux

        # NOTE dissipative cell face fluxes require edge halos
        self.fill_edge_halos_material = any((
            is_viscous_flux,
            is_heat_flux,
        ))

        if equation_information.is_solid_levelset:
            # NOTE dissipative fluid-solid interface fluxes require vertex halos
            self.fill_vertex_halos_material = any((is_viscous_flux,is_heat_flux))
        elif equation_type == "TWO-PHASE-LS":
            # NOTE dissipative fluid-fluid interface fluxes require vertex halos,
            # if interpolation based method is used
            interface_flux_method = numerical_setup.levelset.interface_flux.method
            if interface_flux_method == "INTERPOLATION" and any((is_viscous_flux,is_heat_flux)):
                self.fill_vertex_halos_material = True
            else:
                self.fill_vertex_halos_material = False
        else:
            self.fill_vertex_halos_material = False
    

    def perform_halo_update_material(
            self,
            primitives: Array,
            physical_simulation_time: float,
            fill_edge_halos: bool,
            fill_vertex_halos: bool,
            conservatives: Array = None,
            fill_face_halos: bool = True,
            ml_setup: MachineLearningSetup = MachineLearningSetup()
        ) -> Tuple[Array, Array]:
        """Performs a halo update for the material
        field buffers.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        if conservatives is not None:
            compute_conservatives = True
        else:
            compute_conservatives = False

        if compute_conservatives:
            if self.is_parallel:
                if fill_face_halos:
                    primitives, conservatives = self.halo_communication_material.face_halo_update(
                        primitives, conservatives)
                    primitives, conservatives = self.boundary_condition_material.face_halo_update(
                        primitives, physical_simulation_time, conservatives, ml_setup)
                if self.dim > 1 and fill_edge_halos:
                    primitives, conservatives = self.halo_communication_material.edge_halo_update(
                        primitives, conservatives)
                    primitives, conservatives = self.boundary_condition_material.edge_halo_update(
                        primitives, conservatives)
                if self.dim == 3 and fill_vertex_halos:
                    primitives, conservatives = self.halo_communication_material.vertex_halo_update(
                        primitives, conservatives)
                    primitives, conservatives = self.boundary_condition_material.vertex_halo_update(
                        primitives, conservatives)
            else:
                if fill_face_halos:
                    primitives, conservatives = self.boundary_condition_material.face_halo_update(
                        primitives, physical_simulation_time, conservatives, ml_setup)
                if self.dim > 1 and fill_edge_halos:
                    primitives, conservatives = self.boundary_condition_material.edge_halo_update(
                        primitives, conservatives)
                if self.dim == 3 and fill_vertex_halos:
                    primitives, conservatives = self.boundary_condition_material.vertex_halo_update(
                        primitives, conservatives)

        else:
            if self.is_parallel:
                if fill_face_halos:
                    primitives = self.halo_communication_material.face_halo_update(
                        primitives)
                    primitives = self.boundary_condition_material.face_halo_update(
                        primitives, physical_simulation_time, ml_setup=ml_setup)
                if self.dim > 1 and fill_edge_halos:
                    primitives = self.halo_communication_material.edge_halo_update(
                        primitives)
                    primitives = self.boundary_condition_material.edge_halo_update(
                        primitives)
                if self.dim == 3 and fill_vertex_halos:
                    primitives = self.halo_communication_material.vertex_halo_update(
                        primitives)
                    primitives = self.boundary_condition_material.vertex_halo_update(
                        primitives)

            else:
                if fill_face_halos:
                    primitives = self.boundary_condition_material.face_halo_update(
                        primitives, physical_simulation_time, ml_setup=ml_setup)
                if self.dim > 1 and fill_edge_halos:
                    primitives = self.boundary_condition_material.edge_halo_update(
                        primitives)
                if self.dim == 3 and fill_vertex_halos:
                    primitives = self.boundary_condition_material.vertex_halo_update(
                        primitives)

        if compute_conservatives:
            return primitives, conservatives
        else:
            return primitives

    def perform_outer_halo_update_temperature(
            self,
            temperature: Array,
            physical_simulation_time: float
            ) -> Array:
        """Updates the outer halos of the
        temperature buffer

        :param temperature: _description_
        :type temperature: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """
        temperature = self.boundary_condition_material.face_halo_update_temperature(
            temperature, physical_simulation_time)
        return temperature

    def perform_halo_update_levelset(
            self,
            levelset: Array,
            fill_edge_halos: bool,
            fill_vertex_halos: bool,
            is_geometry_halos = False,
            fill_face_halos = True
            ) -> Array:
        """Halo update for levelset related fields, i.e.,
        levelset and interface quantities.

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: Array
        """

        if self.is_parallel:
            if fill_face_halos:
                levelset = self.halo_communication_levelset.face_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
                levelset = self.boundary_condition_levelset.face_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
            if self.dim > 1 and fill_edge_halos:
                levelset = self.halo_communication_levelset.edge_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
                levelset = self.boundary_condition_levelset.edge_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
            if self.dim == 3 and fill_vertex_halos:
                levelset = self.halo_communication_levelset.vertex_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
                levelset = self.boundary_condition_levelset.vertex_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
        else:
            if fill_face_halos:
                levelset = self.boundary_condition_levelset.face_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
            if self.dim > 1 and fill_edge_halos:
                levelset = self.boundary_condition_levelset.edge_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
            if self.dim == 3 and fill_vertex_halos:
                levelset = self.boundary_condition_levelset.vertex_halo_update(
                    levelset, is_geometry_halos=is_geometry_halos)
                
        return levelset

    def perform_halo_update_conservatives_mixing(
            self,
            conservatives: Array,
            ) -> Array:
        """Updates the halo cells of the integrated 
        conservatives buffer. This is required for mixing
        in levelset simulations.

        :param conservatives: _description_
        :type conservatives: Array
        :param volume_fraction: _description_
        :type volume_fraction: Array
        :return: _description_
        :rtype: Array
        """
        if self.is_parallel:
            conservatives = self.halo_communication_material.face_halo_update(conservatives)
            conservatives = self.boundary_condition_conservative_mixing.face_halo_update(conservatives)
            if self.dim > 1 and self.mixing_targets_conservatives > 1:
                conservatives = self.halo_communication_material.edge_halo_update(conservatives)
                conservatives = self.boundary_condition_conservative_mixing.edge_halo_update(conservatives)
            if self.dim == 3 and self.mixing_targets_conservatives == 3:
                conservatives = self.halo_communication_material.vertex_halo_update(conservatives)
                conservatives = self.boundary_condition_conservative_mixing.vertex_halo_update(conservatives)
        else:
            conservatives = self.boundary_condition_conservative_mixing.face_halo_update(conservatives)
            if self.dim > 1 and self.mixing_targets_conservatives > 1:
                conservatives = self.boundary_condition_conservative_mixing.edge_halo_update(conservatives)
            if self.dim == 3 and self.mixing_targets_conservatives == 3:
                conservatives = self.boundary_condition_conservative_mixing.vertex_halo_update(conservatives)
        return conservatives
 
    def perform_halo_update_diffuse_curvature(
            self,
            curvature: Array,
            ) -> Array:
        """Updates the halo cells for the curvature 
        buffer during the iterative correction procedure
        for diffuse interface simulations.
        """

        if self.is_parallel:
            curvature = self.halo_communication_material.face_halo_update(
                curvature, is_geometry_halos=True)
            curvature = self.boundary_condition_curvature.face_halo_update(
                curvature)
            if self.dim > 1:
                curvature = self.halo_communication_material.edge_halo_update(
                    curvature, is_geometry_halos=True)
                curvature = self.boundary_condition_curvature.edge_halo_update(
                    curvature)
            if self.dim == 3:
                curvature = self.halo_communication_material.vertex_halo_update(
                    curvature, is_geometry_halos=True)
                curvature = self.boundary_condition_curvature.vertex_halo_update(
                    curvature)
        else:
            curvature = self.boundary_condition_curvature.face_halo_update(
                curvature)
            if self.dim > 1:
                curvature = self.boundary_condition_curvature.edge_halo_update(
                    curvature)
            if self.dim == 3:
                curvature = self.boundary_condition_curvature.vertex_halo_update(
                    curvature)

        return curvature



    def perform_halo_update_solids(
            self,
            solid_temperature: Array,
            physical_simulation_time: float,
            fill_edge_halos: bool,
            fill_vertex_halos: bool, 
            solid_energy: Array = None,
            fill_face_halos: bool = True
            ) -> Array:
        """Halo update for solid temperature.

        :param solid_temperature: _description_
        :type solid_temperature: Array
        :return: _description_
        :rtype: Array
        """

        if solid_energy is not None:
            if self.is_parallel:
                if fill_face_halos:
                    solid_temperature, solid_energy = self.halo_communication_solids.face_halo_update(
                        solid_temperature, solid_energy)
                    solid_temperature, solid_energy = self.boundary_condition_solids.face_halo_update(
                        solid_temperature, physical_simulation_time, solid_energy)
                if self.dim > 1 and fill_edge_halos:
                    solid_temperature, solid_energy = self.halo_communication_solids.edge_halo_update(
                        solid_temperature, solid_energy)
                    solid_temperature, solid_energy = self.boundary_condition_solids.edge_halo_update(
                        solid_temperature, solid_energy)
                if self.dim == 3 and fill_vertex_halos:
                    solid_temperature, solid_energy = self.halo_communication_solids.vertex_halo_update(
                        solid_temperature, solid_energy)
                    solid_temperature, solid_energy = self.boundary_condition_solids.vertex_halo_update(
                        solid_temperature, solid_energy)
            else:
                if fill_face_halos:
                    solid_temperature, solid_energy = self.boundary_condition_solids.face_halo_update(
                        solid_temperature, physical_simulation_time, solid_energy)
                if self.dim > 1 and fill_edge_halos:
                    solid_temperature, solid_energy = self.boundary_condition_solids.edge_halo_update(
                        solid_temperature, solid_energy)
                if self.dim == 3 and fill_vertex_halos:
                    solid_temperature, solid_energy = self.boundary_condition_solids.vertex_halo_update(
                        solid_temperature, solid_energy)
        else:
            if self.is_parallel:
                if fill_face_halos:
                    solid_temperature = self.halo_communication_solids.face_halo_update(
                        solid_temperature)
                    solid_temperature = self.boundary_condition_solids.face_halo_update(
                        solid_temperature, physical_simulation_time)
                if self.dim > 1 and fill_edge_halos:
                    solid_temperature = self.halo_communication_solids.edge_halo_update(
                        solid_temperature)
                    solid_temperature = self.boundary_condition_solids.edge_halo_update(
                        solid_temperature)
                if self.dim == 3 and fill_vertex_halos:
                    solid_temperature = self.halo_communication_solids.vertex_halo_update(
                        solid_temperature)
                    solid_temperature = self.boundary_condition_solids.vertex_halo_update(
                        solid_temperature)
            else:
                if fill_face_halos:
                    solid_temperature = self.boundary_condition_solids.face_halo_update(
                        solid_temperature, physical_simulation_time)
                if self.dim > 1 and fill_edge_halos:
                    solid_temperature = self.boundary_condition_solids.edge_halo_update(
                        solid_temperature)
                if self.dim == 3 and fill_vertex_halos:
                    solid_temperature = self.boundary_condition_solids.vertex_halo_update(
                        solid_temperature)


        if solid_energy is not None:
            return solid_temperature, solid_energy
        else:
            return solid_temperature
    

    def perform_halo_update_solids_mixing(
            self,
            solid_energy: Array,
            ) -> Array:
        if self.is_parallel:
            solid_energy = self.halo_communication_solids.face_halo_update(solid_energy)
            solid_energy = self.boundary_condition_solids_mixing.face_halo_update(solid_energy)
            if self.dim > 1 and self.mixing_targets_solids > 1:
                solid_energy = self.halo_communication_solids.edge_halo_update(solid_energy)
                solid_energy = self.boundary_condition_solids_mixing.edge_halo_update(solid_energy)
            if self.dim == 3 and self.mixing_targets_solids == 3:
                solid_energy = self.halo_communication_solids.vertex_halo_update(solid_energy)
                solid_energy = self.boundary_condition_solids_mixing.vertex_halo_update(solid_energy)
        else:
            solid_energy = self.boundary_condition_solids_mixing.face_halo_update(solid_energy)
            if self.dim > 1 and self.mixing_targets_solids > 1:
                solid_energy = self.boundary_condition_solids_mixing.edge_halo_update(solid_energy)
            if self.dim == 3 and self.mixing_targets_solids == 3:
                solid_energy = self.boundary_condition_solids_mixing.vertex_halo_update(solid_energy)
        return solid_energy
    

    def get_cell_sizes_with_halos(self) -> Tuple[Array]:
        """Generates cell sizes with
        halos.

        :return: _description_
        :rtype: Tuple
        """

        nh = self.domain_information.nh_conservatives
        number_of_cells = self.domain_information.device_number_of_cells
        global_cell_sizes = self.domain_information.get_global_cell_sizes()
        local_cell_sizes = self.domain_information.get_local_cell_sizes()
        is_parallel = self.domain_information.is_parallel
        active_axes_indices = self.domain_information.active_axes_indices
        is_mesh_stretching = self.domain_information.is_mesh_stretching

        def update_cell_sizes(
                cell_sizes_xi: Array,
                axis_index: int,
                ) -> Array:
            nxi = number_of_cells[axis_index]
            cell_sizes_xi = cell_sizes_xi.flatten()
            dxi_with_halos = jnp.zeros(nxi + 2*nh)
            dxi_with_halos = dxi_with_halos.at[nh:-nh].set(cell_sizes_xi)
            if is_parallel:
                dxi_with_halos = self.halo_communication_mesh.face_halo_update(dxi_with_halos, axis_index)
            dxi_with_halos = self.boundary_condition_mesh.face_halo_update(dxi_with_halos, axis_index)
            shape = np.roll(np.array([-1,1,1]), axis_index)
            dxi_with_halos = dxi_with_halos.reshape(shape)
            if is_parallel:
                dxi_with_halos = jax.lax.all_gather(dxi_with_halos, axis_name="i")
            return dxi_with_halos

        cell_sizes_with_halos = [dxi for dxi in global_cell_sizes]
        for axis_index in active_axes_indices:
            dxi = local_cell_sizes[axis_index]
            # NOTE cell sizes halos only required for active mesh stretching, else cell size shape is (1,1,1)
            if is_mesh_stretching[axis_index]:
                if is_parallel:
                    dxi_with_halos = jax.pmap(update_cell_sizes, axis_name="i",
                                            static_broadcasted_argnums=(1,), out_axes=(None),
                                            in_axes=(0,None))(dxi, axis_index)
                else:
                    dxi_with_halos = update_cell_sizes(dxi, axis_index)
                cell_sizes_with_halos[axis_index] = dxi_with_halos
        return tuple(cell_sizes_with_halos)
    

    def get_cell_centers_with_halos(self) -> Tuple[Array]:
        """Generates cell centers with halos.

        :return: _description_
        :rtype: Tuple[Array]
        """

        nh = self.domain_information.nh_conservatives
        number_of_cells = self.domain_information.device_number_of_cells
        global_cell_centers = self.domain_information.get_global_cell_centers()
        local_cell_centers = self.domain_information.get_local_cell_centers()
        active_axes_indices = self.domain_information.active_axes_indices
        is_parallel = self.domain_information.is_parallel

        def update_cell_centers(
                cell_centers_xi: Array,
                axis_index: int
                ) -> Tuple[Array, Array]:
            nxi = number_of_cells[axis_index]
            cell_centers_xi = cell_centers_xi.flatten()
            xi_with_halos = jnp.zeros(nxi + 2*nh)
            xi_with_halos = xi_with_halos.at[nh:-nh].set(cell_centers_xi)
            xi_with_halos = self.boundary_condition_mesh.face_halo_update(xi_with_halos, axis_index, "cell_centers")
            xi_diff = jnp.diff(xi_with_halos[nh-1:-nh+1])
            shape = np.roll(np.array([-1,1,1]), axis_index)
            xi_with_halos = xi_with_halos.reshape(shape)
            xi_diff = xi_diff.reshape(shape)
            if is_parallel:
                xi_with_halos = jax.lax.all_gather(xi_with_halos, axis_name="i")
                xi_diff = jax.lax.all_gather(xi_diff, axis_name="i")
            return xi_with_halos, xi_diff

        cell_centers_with_halos = [xi for xi in global_cell_centers]
        cell_centers_difference = [xi for xi in global_cell_centers]
        for axis_index in active_axes_indices:
            xi = local_cell_centers[axis_index]
            if is_parallel:
                xi_with_halos, xi_diff = jax.pmap(update_cell_centers, axis_name="i",
                                        static_broadcasted_argnums=(1,), out_axes=(None,None),
                                        in_axes=(0,None))(xi, axis_index)
            else:
                xi_with_halos, xi_diff = update_cell_centers(xi, axis_index)
            cell_centers_with_halos[axis_index] = xi_with_halos
            cell_centers_difference[axis_index] = xi_diff

        return tuple(cell_centers_with_halos), tuple(cell_centers_difference)