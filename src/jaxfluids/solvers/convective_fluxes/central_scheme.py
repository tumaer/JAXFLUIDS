from __future__ import annotations
from typing import Dict, Union, TYPE_CHECKING, Tuple, List
import copy

import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.solvers.riemann_solvers.eigendecomposition import Eigendecomposition
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.stencils.reconstruction.shock_capturing.weno import WENO1
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.split_reconstruction import SplitReconstruction
from jaxfluids.config import precision

if TYPE_CHECKING:
    from jaxfluids.data_types.ml_buffers import MachineLearningSetup
    from jaxfluids.data_types.numerical_setup.conservatives import ConvectiveFluxesSetup, HighOrderGodunovSetup
    from jaxfluids.data_types.numerical_setup.conservatives import PositivitySetup
    from jaxfluids.data_types.numerical_setup.diffuse_interface import DiffuseInterfaceSetup
    from jaxfluids.diffuse_interface.diffuse_interface_handler import DiffuseInterfaceHandler
    from jaxfluids.solvers.positivity.positivity_handler import PositivityHandler

Array = jax.Array

class CentralScheme(ConvectiveFluxSolver):
    def __init__(
            self,
            convective_fluxes_setup: ConvectiveFluxesSetup,
            material_manager: MaterialManager,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            **kwargs
            ) -> None:

        super(CentralScheme, self).__init__(
            convective_fluxes_setup, material_manager, domain_information, equation_manager)

        self.central_scheme_setup = convective_fluxes_setup.central

        reconstruction_stencil = self.central_scheme_setup.reconstruction_stencil

        self.reconstruction_stencil: SpatialReconstruction = reconstruction_stencil(
            domain_information.nh_conservatives,
            domain_information.inactive_axes,
            )

    def compute_flux_xi(
            self,
            primitives: Array,
            conservatives: Array,
            axis: int,
            curvature: Array = None,
            volume_fraction: Array = None,
            apertures: Tuple[Array] = None,
            ml_setup: MachineLearningSetup = None,
        ) -> Tuple[Array, Array, Array, int]:
        """Computes the numerical flux in a specified spatial direction.

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param conservatives: Buffer of conservative variables.
        :type conservatives: Array
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis direction.
        :rtype: Array
        """

        split_form = self.central_scheme_setup.split_form
        reconstruction_variable = self.central_scheme_setup.reconstruction_variable

        if split_form is None:
            
            if reconstruction_variable == "PRIMITIVE":

                primitives_xi = self.reconstruction_stencil.reconstruct_xi(primitives, axis)
                conservatives_xi = self.equation_manager.get_conservatives_from_primitives(primitives_xi)
                fluxes_xi = self.equation_manager.get_fluxes_xi(primitives_xi, conservatives_xi, axis)

            elif reconstruction_variable == "CONSERVATIVE":

                conservatives_xi = self.reconstruction_stencil.reconstruct_xi(conservatives, axis)
                primitives_xi = self.equation_manager.get_primitives_from_conservatives(primitives_xi)
                fluxes_xi = self.equation_manager.get_fluxes_xi(primitives_xi, conservatives_xi, axis)

            elif reconstruction_variable == "FLUX":

                fluxes_xi = self.equation_manager.get_fluxes_xi(primitives, conservatives, axis)
                fluxes_xi = self.reconstruction_stencil.reconstruct_xi(fluxes_xi, axis)

        else:

            fluxes_xi = self.equation_manager.get_fluxes_xi(primitives, conservatives, axis)

            nh = self.domain_information.nh_conservatives
            nhx,nhy,nhz = self.domain_information.domain_slices_conservatives
            s0_tuple = (
                jnp.s_[nh-1:-nh,nhy,nhz],
                jnp.s_[nhx,nh-1:-nh,nhz],
                jnp.s_[nhx,nhy,nh-1:-nh]
                )
            s1_tuple = (
                jnp.s_[nh:-nh+1,nhy,nhz],
                jnp.s_[nhx,nh:-nh+1,nhz],
                jnp.s_[nhx,nhy,nh:-nh+1]
                )

            s0 = s0_tuple[axis]
            s1 = s1_tuple[axis]

            rho = primitives[0]
            u = primitives[1]
            v = primitives[2]
            w = primitives[3]
            p = primitives[4]

            rho0 = rho[s0]
            u0 = u[s0]
            v0 = v[s0]
            w0 = w[s0]
            p0 = p[s0]

            rho1 = rho[s1]
            u1 = u[s1]
            v1 = v[s1]
            w1 = w[s1]
            p1 = p[s1]

            rhou = conservatives[1]
            rhov = conservatives[2]
            rhow = conservatives[3]
            E = conservatives[4]

            rhou0 = rhou[s0]
            rhov0 = rhov[s0]
            rhow0 = rhow[s0]
            E0 = E[s0]

            rhou1 = rhou[s1]
            rhov1 = rhov[s1]
            rhow1 = rhow[s1]
            E1 = E[s1]

            if split_form == "FEIEREISEN":

                factor = 1.0/4.0

                if axis == 0:
                    fluxes_rho_xi = 1.0/2.0 * (rhou0 + rhou1)
                    fluxes_u_xi = factor * (rhou0 + rhou1) * (u0 + u1) + 1.0/2.0 * (p0 + p1)
                    fluxes_v_xi = factor * (rhou0 + rhou1) * (v0 + v1)
                    fluxes_w_xi = factor * (rhou0 + rhou1) * (w0 + w1)
                    fluxes_E_xi = 1.0/2.0 * (u0*(E0+p0) + u1*(E1+p1))

                elif axis == 1:
                    fluxes_rho_xi = 1.0/2.0 * (rhov0 + rhov1)
                    fluxes_u_xi = factor * (rhov0 + rhov1) * (u0 + u1) 
                    fluxes_v_xi = factor * (rhov0 + rhov1) * (v0 + v1) + 1.0/2.0 * (p0 + p1)
                    fluxes_w_xi = factor * (rhov0 + rhov1) * (w0 + w1)
                    fluxes_E_xi = 1.0/2.0 * (v0*(E0+p0) + v1*(E1+p1))
                else:
                    fluxes_rho_xi = 1.0/2.0 * (rhow0 + rhow1)
                    fluxes_u_xi = factor * (rhow0 + rhow1) * (u0 + u1)
                    fluxes_v_xi = factor * (rhow0 + rhow1) * (v0 + v1)
                    fluxes_w_xi = factor * (rhow0 + rhow1) * (w0 + w1) + 1.0/2.0 * (p0 + p1)
                    fluxes_E_xi = 1.0/2.0 * (w0*(E0+p0) + w1*(E1+p1))

            elif split_form == "BLAISDELL":

                factor = 1.0/4.0

                if axis == 0:
                    fluxes_rho_xi = factor * (rho0 + rho1) * (u0 + u1)
                    fluxes_u_xi = factor * (rhou0 + rhou1) * (u0 + u1) + 1.0/2.0 * (p0 + p1)
                    fluxes_v_xi = factor * (rhov0 + rhov1) * (u0 + u1)
                    fluxes_w_xi = factor * (rhow0 + rhow1) * (u0 + u1)
                    fluxes_E_xi = factor * (E0 + E1) * (u0 + u1) + 1.0/2.0 * (p0*u0 + p1*u1) 
                elif axis == 1:
                    fluxes_rho_xi = factor * (rho0 + rho1) * (v0 + v1)
                    fluxes_u_xi = factor * (rhou0 + rhou1) * (v0 + v1) 
                    fluxes_v_xi = factor * (rhov0 + rhov1) * (v0 + v1) + 1.0/2.0 * (p0 + p1)
                    fluxes_w_xi = factor * (rhow0 + rhow1) * (v0 + v1)
                    fluxes_E_xi = factor * (E0 + E1) * (v0 + v1) + 1.0/2.0 * (p0*v0 + p1*v1) 
                else:
                    fluxes_rho_xi = factor * (rho0 + rho1) * (w0 + w1)
                    fluxes_u_xi = factor * (rhou0 + rhou1) * (w0 + w1)
                    fluxes_v_xi = factor * (rhov0 + rhov1) * (w0 + w1)
                    fluxes_w_xi = factor * (rhow0 + rhow1) * (w0 + w1) + 1.0/2.0 * (p0 + p1)
                    fluxes_E_xi = factor * (E0 + E1) * (w0 + w1) + 1.0/2.0 * (p0*w0 + p1*w1) 

            elif split_form == "KENNEDY":
                
                factor = 1.0/8.0
                e = E/rho
                e0 = e[s0]
                e1 = e[s1]

                if axis == 0:
                    fluxes_rho_xi = 2 * factor * (rho0 + rho1) * (u0 + u1)
                    fluxes_u_xi = factor * (rho0 + rho1) * (u0 + u1) * (u0 + u1) + 1.0/2.0 * (p0 + p1)
                    fluxes_v_xi = factor * (rho0 + rho1) * (v0 + v1) * (u0 + u1)
                    fluxes_w_xi = factor * (rho0 + rho1) * (w0 + w1) * (u0 + u1)
                    fluxes_E_xi = factor * (rho0 + rho1) * (e0 + e1) * (u0 + u1) + 1.0/2.0 * (p0*u0 + p1*u1) 
                    # fluxes_E_xi = factor * (rho0 + rho1) * (e0 + e1) * (u0 + u1) + 1.0/4.0 * (p0 + p1) * (u0 + u1) 

                elif axis == 1:
                    fluxes_rho_xi = 2 * factor * (rho0 + rho1) * (v0 + v1)
                    fluxes_u_xi = factor * (rho0 + rho1) * (u0 + u1) * (v0 + v1) 
                    fluxes_v_xi = factor * (rho0 + rho1) * (v0 + v1) * (v0 + v1) + 1.0/2.0 * (p0 + p1)
                    fluxes_w_xi = factor * (rho0 + rho1) * (w0 + w1) * (v0 + v1)
                    fluxes_E_xi = factor * (rho0 + rho1) * (e0 + e1) * (v0 + v1) + 1.0/2.0 * (p0*v0 + p1*v1) 
                    # fluxes_E_xi = factor * (rho0 + rho1) * (e0 + e1) * (v0 + v1) + 1.0/4.0 * (p0 + p1) * (v0 + v1) 

                else:
                    fluxes_rho_xi = 2 * factor * (rho0 + rho1) * (w0 + w1)
                    fluxes_u_xi = factor * (rho0 + rho1) * (u0 + u1) * (w0 + w1)
                    fluxes_v_xi = factor * (rho0 + rho1) * (v0 + v1) * (w0 + w1)
                    fluxes_w_xi = factor * (rho0 + rho1) * (w0 + w1) * (w0 + w1) + 1.0/2.0 * (p0 + p1)
                    fluxes_E_xi = factor * (rho0 + rho1) * (e0 + e1) * (w0 + w1) + 1.0/2.0 * (p0*w0 + p1*w1)
                    # fluxes_E_xi = factor * (rho0 + rho1) * (e0 + e1) * (w0 + w1) + 1.0/4.0 * (p0 + p1) * (w0 + w1) 
            
            else:
                raise NotImplementedError

            fluxes_xi = jnp.stack([fluxes_rho_xi, fluxes_u_xi, fluxes_v_xi,
                                   fluxes_w_xi, fluxes_E_xi])

        return fluxes_xi, None, None, None, None