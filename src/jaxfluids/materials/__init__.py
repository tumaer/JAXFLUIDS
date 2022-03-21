from jaxfluids.materials.ideal_gas import IdealGas
from jaxfluids.materials.safe_ideal_gas import SafeIdealGas
from jaxfluids.materials.stiffened_gas import StiffenedGas
from jaxfluids.materials.tait import Tait

DICT_MATERIAL = {
    'IdealGas': IdealGas,
    'SafeIdealGas': SafeIdealGas,
    'StiffenedGas': StiffenedGas,
    'Tait': Tait,
}