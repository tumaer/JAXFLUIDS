from jaxfluids.materials.single_materials.ideal_gas import IdealGas
from jaxfluids.materials.single_materials.safe_ideal_gas import SafeIdealGas
from jaxfluids.materials.single_materials.stiffened_gas import StiffenedGas
from jaxfluids.materials.single_materials.stiffened_gas_complete import StiffenedGasComplete
from jaxfluids.materials.single_materials.tait import Tait

DICT_MATERIAL = {
    'IdealGas': IdealGas,
    'SafeIdealGas': SafeIdealGas,
    'StiffenedGas': StiffenedGas,
    'StiffenedGasComplete': StiffenedGasComplete,
    'Tait': Tait,
}