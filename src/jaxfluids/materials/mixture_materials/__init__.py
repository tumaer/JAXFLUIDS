from jaxfluids.materials.mixture_materials.diffuse_mixture import DiffuseMixture
from jaxfluids.materials.mixture_materials.diffuse_mixture_five_equation import DiffuseFiveEquationMixture
from jaxfluids.materials.mixture_materials.levelset_mixture import LevelsetMixture

DICT_MIXTURE = {
    "DiffuseMixture": DiffuseMixture,
    "DiffuseFiveEquationMixture": DiffuseFiveEquationMixture,
    "LevelSetMixture": LevelsetMixture,
}