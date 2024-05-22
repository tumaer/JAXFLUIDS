from collections import namedtuple
from typing import NamedTuple, Callable, Dict

from jaxfluids.data_types.case_setup.general import GeneralSetup
from jaxfluids.data_types.case_setup.restart import RestartSetup
from jaxfluids.data_types.case_setup.domain import DomainSetup
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionSetup
from jaxfluids.data_types.case_setup.initial_conditions import InitialConditionSetup
from jaxfluids.data_types.case_setup.material_properties import MaterialManagerSetup
from jaxfluids.data_types.case_setup.solid_properties import SolidPropertiesSetup
from jaxfluids.data_types.case_setup.forcings import ForcingSetup
from jaxfluids.data_types.case_setup.output import OutputQuantitiesSetup

def GetPrimitivesCallable(
        primitives_callable_dict: Dict
        ):
    fields = primitives_callable_dict.keys()
    PrimitivesCallable = namedtuple("PrimitivesCallable", fields)
    primitives_callable = PrimitivesCallable(**primitives_callable_dict)
    return primitives_callable

class CaseSetup(NamedTuple):
    general_setup: GeneralSetup
    restart_setup: RestartSetup
    domain_setup: DomainSetup
    boundary_condition_setup: BoundaryConditionSetup
    initial_condition_setup: InitialConditionSetup
    material_manager_setup: MaterialManagerSetup
    solid_properties_setup: SolidPropertiesSetup
    forcing_setup: ForcingSetup
    output_quantities_setup: OutputQuantitiesSetup

