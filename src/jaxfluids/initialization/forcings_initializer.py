from typing import Union, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import h5py
import warnings

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.turb.statistics.utilities import energy_spectrum_physical, energy_spectrum_physical_parallel
from jaxfluids.data_types.buffers import ForcingParameters, \
    MassFlowControllerParameters, MaterialFieldBuffers
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.data_types.case_setup import RestartSetup

class ForcingsInitializer:
    """The ForcingsInitializer implements functionality
    to create initial buffers for the conservative and
    primitive variables.
    """

    def __init__(
            self,
            numerical_setup: NumericalSetup,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            restart_setup: RestartSetup
            ) -> None:

        self.domain_information = domain_information
        self.equation_information = material_manager.equation_information
        self.is_turb_hit_forcing = numerical_setup.active_forcings.is_turb_hit_forcing
        self.is_temperature_forcing = numerical_setup.active_forcings.is_temperature_forcing
        self.is_mass_flow_forcing = numerical_setup.active_forcings.is_mass_flow_forcing
        self.is_double_precision = numerical_setup.precision.is_double_precision_compute

        self.is_restart = restart_setup.flag
        self.restart_file_path = restart_setup.file_path

    def initialize(
            self,
            material_fields: MaterialFieldBuffers
            ) -> ForcingParameters:
        """Wrapper function that initializes the forcing either
        1) from a restart file
        2) from the initial condition

        :param material_fields: _description_
        :type material_fields: MaterialFieldBuffers
        :return: _description_
        :rtype: ForcingParameters
        """

        if self.is_restart:
            forcings = self.from_restart_file(material_fields)
        else:
            forcings = self.from_initial_condition(material_fields)

        return forcings

    def from_restart_file(
            self,
            material_fields: MaterialFieldBuffers
            ) -> ForcingParameters:
        
        # LOAD H5FILE
        h5file = h5py.File(self.restart_file_path, "r")
        available_quantities = h5file["metadata"]["available_quantities"].keys()

        hit_ek_ref = None
        mass_flow_controller_params = None

        mass_flow_params_default = False
        hit_ek_ref_default = False
            
        # DEFAULT FORCINGS
        if "forcings" not in available_quantities:
            warning_string = ("Restart file %s does not "
            "contain forcings, however, there are active forcings. "
            "Using default values" % self.restart_file_path)
            warnings.warn(warning_string, RuntimeWarning)

            if self.is_mass_flow_forcing:
                mass_flow_params_default = True
            else:
                mass_flow_params_default = False

            if self.is_turb_hit_forcing:
                hit_ek_ref_default = True
            else:
                hit_ek_ref_default = False

        else:
            forcings_restart = h5file["forcings"]
            if self.is_mass_flow_forcing:
                if "mass_flow" not in forcings_restart.keys():
                    warning_string = ("Restart file %s does not "
                    "contain mass flow forcing, however, "
                    "mass flow forcing is active. "
                    "Defaulting PID controller " 
                    "parameters to 0.0" % self.restart_file_path)
                    warnings.warn(warning_string, RuntimeWarning)
                    mass_flow_params_default = True 
                else:
                    mass_flow_params_default = False
                    mass_flow_controller_params = MassFlowControllerParameters(
                        forcings_restart["mass_flow"]["PID_e_new"][()],
                        forcings_restart["mass_flow"]["PID_e_int"][()])

            if self.is_turb_hit_forcing:
                if "turb_hit" not in forcings_restart.keys():
                    warning_string = ("Restart file %s does not "
                    "contain hit forcing, however, hit forcing is active. "
                    "Computing reference energy spectrum from restart velocity "
                    "field." % self.restart_file_path)
                    warnings.warn(warning_string, RuntimeWarning)
                    hit_ek_ref_default = True
                else:
                    hit_ek_ref_default = False
                    hit_ek_ref = forcings_restart["turb_hit"]["ek_ref"][:]

        if mass_flow_params_default:
            mass_flow_controller_params = MassFlowControllerParameters(
                0.0, 0.0)

        if hit_ek_ref_default:
            def compute_energy_spectrum_wrapper(primitives: Array):
                nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
                velocity = primitives[1:4,...,nhx,nhy,nhz]
                energy_spec = energy_spectrum_physical(velocity, multiplicative_factor=0.5)
                return energy_spec
            if self.domain_information.is_parallel:
                # TODO multi-host
                if self.domain_information.is_multihost:
                    warning_string = ("compute_energy_spectrum_wrapper is not "
                        "tested in multi-host settings.")
                    warnings.warn(warning_string, RuntimeWarning)

                hit_ek_ref = jax.pmap(compute_energy_spectrum_wrapper, out_axes=(None),
                                        axis_name="i")(material_fields.primitives)
            else:
                hit_ek_ref = compute_energy_spectrum_wrapper(material_fields.primitives)
                
        forcing_parameters = ForcingParameters(
            mass_flow_controller_params, hit_ek_ref)

        return forcing_parameters
        

    def from_initial_condition(
            self,
            material_fields: MaterialFieldBuffers
            ) -> ForcingParameters:
        """Creates the forcings 

        :param material_fields: _description_
        :type material_fields: MaterialFieldBuffers[str, Array]
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: Dict
        """

        # DOMAIN INFORMATION
        is_parallel = self.domain_information.is_parallel
        split_factors = self.domain_information.split_factors
        number_of_cells = self.domain_information.global_number_of_cells

        if self.is_mass_flow_forcing:
            mass_flow_controller_params = MassFlowControllerParameters(
                0.0, 0.0)
        else:
            mass_flow_controller_params = None

        if self.is_turb_hit_forcing:

            primitives = material_fields.primitives

            def func(primitives):
                """Wrapper to compute the energy spectrum.

                :param primitives: _description_
                :type primitives: _type_
                :return: _description_
                :rtype: _type_
                """
                velocity_slices = self.equation_information.velocity_slices
                domain_slices = self.domain_information.domain_slices_conservatives
                slice_object = tuple([velocity_slices,]) + domain_slices
                velocity = primitives[slice_object]
                if is_parallel:
                    energy_spec = energy_spectrum_physical_parallel(
                        velocity, split_factors, multiplicative_factor=0.5)
                else:
                    energy_spec = energy_spectrum_physical(
                        velocity, multiplicative_factor=0.5)
                return energy_spec
            
            if is_parallel:
                hit_ek_ref = jax.pmap(func, axis_name="i", out_axes=(None))(primitives)
            else:
                hit_ek_ref = func(primitives)

        else:
            hit_ek_ref = None

        forcing_parameters = ForcingParameters(
            mass_flow_controller_params, hit_ek_ref)

        return forcing_parameters

