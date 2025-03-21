from jaxfluids.levelset.reinitialization.godunov_hamiltonian_reinitializer import GodunovHamiltonianReinitializer
from jaxfluids.levelset.reinitialization.russo_reinitializer import RussoReinitializer

DICT_LEVELSET_REINITIALIZER = {
    "GODUNOVHAMILTONIAN": GodunovHamiltonianReinitializer,
    "RUSSO": RussoReinitializer,
}