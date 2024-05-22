import json
import yaml
import os
from typing import Dict, Union, Tuple
import warnings

import jax
import jax.numpy as jnp
from jax import Array, numpy as np

from jaxfluids.config import precision as precision_config
from jaxfluids.data_types.case_setup.nondimensionalization import NondimensionalizationParameters
from jaxfluids.data_types.numerical_setup.precision import PrecisionSetup, Epsilons
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.input.case_setup.case_setup_reader import CaseSetupReader
from jaxfluids.input.numerical_setup.numerical_setup_reader import NumericalSetupReader
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.solvers.convective_fluxes import HighOrderGodunov, FluxSplittingScheme

class InputManager:
    """ The InputManager class reads a case setup
    and a numerical setup for setting up a JAX-FLUIDS simulation. Case setup
    and numerical setup can be provided as either a path to a json file or 
    as a preloaded dictionary. This class performs function transformations
    and sanity checks.
    """

    def __init__(
            self,
            case_setup: Union[str, Dict],
            numerical_setup: Union[str, Dict],
            materials_setup: Union[str, Dict] = None,
            ) -> None:

        # CASE SETUP
        self.case_setup_dict = read_json_setup(case_setup, "case setup")

        # NUMERICAL SETUP
        self.numerical_setup_dict = read_json_setup(numerical_setup, "numerical setup")

        # MATERIALS SETUP
        self.materials_database = None
        if materials_setup:
            self.materials_database = read_json_setup(materials_setup, "materials_setup")
            raise NotImplementedError

        self.unit_handler = get_unit_handler(self.case_setup_dict)

        numerical_setup_reader = NumericalSetupReader(self.unit_handler)
        case_setup_reader = CaseSetupReader(self.unit_handler)

        # READ & INITIALIZE NUMERICAL SETUP AND CASE SETUP
        self.numerical_setup = numerical_setup_reader.initialize_numerical_setup(
            self.numerical_setup_dict)
        self.case_setup, self.equation_information = case_setup_reader.initialize_case_setup(
            self.case_setup_dict, self.numerical_setup)

        self.set_precision_config(self.numerical_setup.precision)

        levelset_model = self.numerical_setup.levelset.model
        diffuse_interface_model = self.numerical_setup.diffuse_interface.model
        if levelset_model:
            nh_geometry = self.numerical_setup.levelset.halo_cells
        elif diffuse_interface_model:
            nh_geometry = self.numerical_setup.diffuse_interface.halo_cells
        else:
            nh_geometry = None

        self.domain_information = DomainInformation(
            domain_setup=self.case_setup.domain_setup,
            nh_conservatives=self.numerical_setup.conservatives.halo_cells,
            nh_geometry=nh_geometry)

        self.material_manager = MaterialManager(
            equation_information=self.equation_information,
            unit_handler=self.unit_handler,
            material_manager_setup=self.case_setup.material_manager_setup)

        self.equation_manager = EquationManager(
            material_manager=self.material_manager,
            equation_information=self.equation_information)

        self.halo_manager = HaloManager(
            numerical_setup=self.numerical_setup,
            domain_information=self.domain_information,
            material_manager=self.material_manager,
            equation_manager=self.equation_manager,
            boundary_conditions_setup=self.case_setup.boundary_condition_setup)

        # WE SET THE CELL SIZES WITH HALOS HERE FOR NOW, SINCE MESH IS STATIC
        cell_sizes_halos = self.halo_manager.get_cell_sizes_with_halos()
        self.domain_information.set_global_cell_sizes_with_halos(cell_sizes_halos)
        cell_centers_halos, cell_centers_difference = self.halo_manager.get_cell_centers_with_halos()
        self.domain_information.set_global_cell_centers_with_halos(cell_centers_halos, cell_centers_difference)
        self.sanity_check()

    def sanity_check(self) -> None:

        # MESH-STRETCHING
        convective_solver = self.numerical_setup.conservatives.convective_fluxes.convective_solver
        is_mesh_streching = self.domain_information.is_mesh_stretching
        if any(is_mesh_streching):
            if convective_solver in (HighOrderGodunov, FluxSplittingScheme):
                reconstruction_stencil = self.numerical_setup.conservatives.convective_fluxes.reconstruction_stencil
                split_reconstruction_setup = self.numerical_setup.conservatives.convective_fluxes.split_reconstruction
                
                if reconstruction_stencil is not None:
                    if not reconstruction_stencil.is_for_adaptive_mesh:
                        warning_string = (
                            "Mesh stretching is active in one of the spatial dimensions. "
                            f"However, the chosen spatial reconstruction stencil "
                            f"'{reconstruction_stencil.__name__}' for convective terms "
                            "is for uniform grids only.")
                        warnings.warn(warning_string, RuntimeWarning)
                
                elif split_reconstruction_setup is not None:
                    for field in split_reconstruction_setup._fields:
                        stencil_field = getattr(split_reconstruction_setup, field)
                        if stencil_field is not None and (not stencil_field.is_for_adaptive_mesh):
                            warning_string = (
                                "Mesh stretching is active in one of the spatial dimensions. "
                                f"However, the chosen spatial reconstruction stencil '{stencil_field.__name__}' "
                                f"for '{field}' for convective terms is for uniform grids only.")
                            warnings.warn(warning_string, RuntimeWarning)
                
                else:
                    raise NotImplementedError

        # ACTIVE LEVELSET, ASPECT RATIO
        levelset_model = self.numerical_setup.levelset.model
        is_mesh_streching = self.domain_information.is_mesh_stretching
        active_axes_indices = self.domain_information.active_axes_indices
        if levelset_model:
            if any(is_mesh_streching):
                warning_string = ("Active levelset and mesh stretching requires the user to ensure "
                                "that the interface is located on the highest resolution with cell size "
                                "aspect ratio unity.")
                warnings.warn(warning_string, RuntimeWarning)
            dx,dy,dz = self.domain_information.get_global_cell_sizes()
            dx_min = jnp.min(dx)
            dy_min = jnp.min(dy)
            dz_min = jnp.min(dz)
            cell_sizes_min = jnp.array([dx_min, dy_min, dz_min])
            cell_sizes_min = cell_sizes_min[np.array(active_axes_indices)]
            smallest_cell_size = jnp.min(cell_sizes_min)
            is_unity_aspect_ratio = jnp.allclose(smallest_cell_size, cell_sizes_min)
            assert_string = ("Cell size aspect ratio on the finest grid must be unity "
                            "when using the level-set model. However, on the finest level, "
                            "minimum and maximum aspect ratios are currently "
                            f"{jnp.min(cell_sizes_min/smallest_cell_size):4.16f} and "
                            f"{jnp.max(cell_sizes_min/smallest_cell_size):4.16f}, respectively.")
            assert is_unity_aspect_ratio, assert_string

    def info(self) -> Tuple[Dict, Dict]:
        """Generates a numerical setup and a case setup dictionary 
        for the logger.
        
        # TODO should be updated

        :return: [description]
        :rtype: Tuple[Dict, Dict]
        """

        # def isnamedtupleinstance(x):
        #     _type = type(x)
        #     bases = _type.__bases__
        #     if len(bases) != 1 or bases[0] != tuple:
        #         return False
        #     fields = getattr(_type, '_fields', None)
        #     if not isinstance(fields, tuple):
        #         return False
        #     return all(type(i)==str for i in fields)

        # def unpack(obj):
        #     if isinstance(obj, dict):
        #         return {key: unpack(value) for key, value in obj.items()}
        #     elif isinstance(obj, list):
        #         return [unpack(value) for value in obj]
        #     elif isnamedtupleinstance(obj):
        #         return {key: unpack(value) for key, value in obj._asdict().items()}
        #     elif isinstance(obj, tuple):
        #         return tuple(unpack(value) for value in obj)
        #     else:
        #         return obj

        # return unpack(self.numerical_setup), unpack(self.case_setup)

        numerical_setup_dict = {}
        for key0, item0 in self.numerical_setup_dict.items():
            if isinstance(item0, dict):
                key0_ = str(key0).replace("_", " ").upper()
                numerical_setup_dict[key0_] = {}
                for key1, item1 in item0.items():
                    key1_ = str(key1).replace("_", " ").upper()
                    if isinstance(item1, dict):
                        numerical_setup_dict[key0_][key1_] = {}
                        for key2, item2 in item1.items():
                            key2_ = str(key2).replace("_", " ").upper()
                            numerical_setup_dict[key0_][key1_][key2_] = item2
                    else:
                        numerical_setup_dict[key0_][key1_] = item1 

        case_setup_dict = {}
        for key0, item0 in self.case_setup_dict.items():
            if isinstance(item0, dict):
                key0_ = str(key0).replace("_", " ").upper()
                case_setup_dict[key0_] = {}
                for key1, item1 in item0.items():
                    key1_ = str(key1).replace("_", " ").upper()
                    if isinstance(item1, dict):
                        case_setup_dict[key0_][key1_] = {}
                        for key2, item2 in item1.items():
                            key2_ = str(key2).replace("_", " ").upper()
                            if isinstance(item2, dict):
                                case_setup_dict[key0_][key1_][key2_] = {}
                                for key3, item3 in item2.items():
                                    key3_ = str(key3).replace("_", " ").upper()
                                    case_setup_dict[key0_][key1_][key2_][key3_] = item3
                            else:
                                case_setup_dict[key0_][key1_][key2_] = item2
                    else:
                        case_setup_dict[key0_][key1_] = item1

        return numerical_setup_dict, case_setup_dict

    def set_precision_config(self, precision_setup: PrecisionSetup):
        """Sets the precision config according to the
        numerical setup json file.

        :param precision_setup: _description_
        :type precision_setup: PrecisionSetup
        """

        if precision_setup.is_double_precision_compute:
            precision_config.enable_double_precision()

        epsilon = precision_setup.epsilon
        spatial_stencil_epsilon = precision_setup.spatial_stencil_epsilon
        fmax = precision_setup.fmax
        smallest_normal = precision_setup.smallest_normal

        if isinstance(epsilon, float):
            precision_config.set_eps(epsilon)
        if isinstance(smallest_normal, float):
            precision_config.set_smallest_normal(smallest_normal)
        if isinstance(fmax, float):
            precision_config.set_fmax(fmax)
        if isinstance(spatial_stencil_epsilon, float):
            precision_config.set_spatial_stencil_eps(spatial_stencil_epsilon)

        interpolation_limiter_epsilons = precision_setup.interpolation_limiter_epsilons
        flux_limiter_epsilons = precision_setup.flux_limiter_epsilons
        thinc_limiter_epsilons = precision_setup.thinc_limiter_epsilons

        def retrieve_values_eps(
                epsilons_tuple_input: Epsilons,
                epsilons_tuple_default: Epsilons,
                ) -> Tuple[float, float, float]:
            """Retrieves the epsilons for the positivity fixes
            from the numerical setup file and sets default values
            if they are not provided.

            :param epsilons_tuple_input: _description_
            :type epsilons_tuple_input: Epsilons
            :param epsilons_tuple_default: _description_
            :type epsilons_tuple_default: Epsilons
            :return: _description_
            :rtype: Tuple[float, float, float]
            """
            epsilons_dict = {}
            for field in epsilons_tuple_input._fields:
                value = getattr(epsilons_tuple_input, field)
                if not isinstance(value, float):
                    value = getattr(epsilons_tuple_default, field)
                epsilons_dict[field] = value
            return epsilons_dict

        interpolation_limiter_epsilons = retrieve_values_eps(
            interpolation_limiter_epsilons,
            precision_config.get_interpolation_limiter_eps())

        flux_limiter_epsilons = retrieve_values_eps(
            flux_limiter_epsilons,
            precision_config.get_flux_limiter_eps())
        
        thinc_limiter_epsilons = retrieve_values_eps(
            thinc_limiter_epsilons,
            precision_config.get_thinc_limiter_eps())
        
        precision_config.set_interpolation_limiter_eps(
            **interpolation_limiter_epsilons)
        precision_config.set_flux_limiter_eps(
            **flux_limiter_epsilons)
        precision_config.set_thinc_limiter_eps(
            **thinc_limiter_epsilons)


def get_unit_handler(case_setup_dict: Dict) -> UnitHandler:
    """Initializes a UniHandler based on
    the nondimensionalization parameters
    provided in the case setup.

    :return: _description_
    :rtype: UnitHandler
    """
    nondim_keys = NondimensionalizationParameters._fields
    default_nondim_params = dict.fromkeys(nondim_keys, 1.0)
    nondim_params = case_setup_dict.get(
        "nondimensionalization_parameters",
        default_nondim_params)
    return UnitHandler(**nondim_params)

def read_json_setup(setup: Union[str, Dict], name: str) -> Dict:
    """Reads the provided setup which can be
    a path to a json file or a dictionary.

    :param setup: _description_
    :type setup: Union[str, Dict]
    :param name: _description_
    :type name: str
    :return: _description_
    :rtype: Dict
    """
    if isinstance(setup, str):
        assert os.path.isfile(setup), (
            "Consistency error reading json setup file. "
            f"{name} file does not exist.")
        setup = json.load(open(setup))

    elif isinstance(setup, dict):
        pass

    else:
        setup_type = type(setup)
        assert False, (f"{name} has to be of type str or dict, "
            f"but is of type {setup_type}.")

    return setup

def read_json_or_yaml_setup(setup: Union[str, Dict], name: str) -> Dict:
    """Reads the provided setup which can be
    a path to a json file, a path to a yaml file
    or a Python dictionary.

    :param setup: _description_
    :type setup: Union[str, Dict]
    :param name: _description_
    :type name: str
    :return: _description_
    :rtype: Dict
    """
    if isinstance(setup, str):
        assert os.path.isfile(setup), (
            "Consistency error reading json setup file. "
            f"{name} file does not exist.")

        if setup.endswith("json"):
            setup = json.load(open(setup))
        elif setup.endswith(("yaml", "yml")):
            setup = yaml.safe_load(open(setup))
        else:
            raise NotImplementedError

    elif isinstance(setup, dict):
        pass

    else:
        setup_type = type(setup)
        assert False, (f"{name} has to be of type str or dict, "
            f"but is of type {setup_type}.")

    return setup
