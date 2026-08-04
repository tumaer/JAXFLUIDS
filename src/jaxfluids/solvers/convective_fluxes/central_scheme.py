from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

if TYPE_CHECKING:
    from jaxfluids.data_types.ml_buffers import MachineLearningSetup
    from jaxfluids.data_types.numerical_setup.conservatives import ConvectiveFluxesSetup

Array = jax.Array

ONE_TWO = 1.0 / 2.0
ONE_FOUR = 1.0 / 4.0
ONE_EIGHT = 1.0 / 8.0


class CentralScheme(ConvectiveFluxSolver):

    def __init__(
            self,
            convective_fluxes_setup: ConvectiveFluxesSetup,
            material_manager: MaterialManager,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            **kwargs,
        ) -> None:

        super(CentralScheme, self).__init__(
            convective_fluxes_setup,
            material_manager,
            domain_information,
            equation_manager,
        )

        self.central_scheme_setup = convective_fluxes_setup.central

        reconstruction_stencil = self.central_scheme_setup.reconstruction_stencil

        self.reconstruction_stencil: SpatialReconstruction = reconstruction_stencil(
            domain_information.nh_conservatives,
            domain_information.inactive_axes,
        )

        nh = self.domain_information.nh_conservatives
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        start = nh - 1 if nh > 1 else None
        stop = -nh
        self.s0_tuple = (
            jnp.s_[start:stop,nhy,nhz],
            jnp.s_[nhx,start:stop,nhz],
            jnp.s_[nhx,nhy,start:stop],
        )
        start = nh
        stop = -nh + 1 if nh > 1 else None
        self.s1_tuple = (
            jnp.s_[start:stop,nhy,nhz],
            jnp.s_[nhx,start:stop,nhz],
            jnp.s_[nhx,nhy,start:stop],
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
        ) -> Tuple[Array, Array, Array, int, int]:
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
                primitives_xi = self.equation_manager.get_primitives_from_conservatives(conservatives_xi)
                fluxes_xi = self.equation_manager.get_fluxes_xi(primitives_xi, conservatives_xi, axis)

            elif reconstruction_variable == "FLUX":

                fluxes_xi = self.equation_manager.get_fluxes_xi(primitives, conservatives, axis)
                fluxes_xi = self.reconstruction_stencil.reconstruct_xi(fluxes_xi, axis)

        else:

            s0 = self.s0_tuple[axis]
            s1 = self.s1_tuple[axis]

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

                p_mean = ONE_TWO * (p0 + p1)

                if axis == 0:
                    fluxes_rho_xi = ONE_TWO * (rhou0 + rhou1)
                    fluxes_u_xi = ONE_FOUR * (rhou0 + rhou1) * (u0 + u1) + p_mean
                    fluxes_v_xi = ONE_FOUR * (rhou0 + rhou1) * (v0 + v1)
                    fluxes_w_xi = ONE_FOUR * (rhou0 + rhou1) * (w0 + w1)
                    fluxes_E_xi = ONE_TWO * (u0 * (E0 + p0) + u1 * (E1 + p1))
                elif axis == 1:
                    fluxes_rho_xi = ONE_TWO * (rhov0 + rhov1)
                    fluxes_u_xi = ONE_FOUR * (rhov0 + rhov1) * (u0 + u1) 
                    fluxes_v_xi = ONE_FOUR * (rhov0 + rhov1) * (v0 + v1) + p_mean
                    fluxes_w_xi = ONE_FOUR * (rhov0 + rhov1) * (w0 + w1)
                    fluxes_E_xi = ONE_TWO * (v0 * (E0 + p0) + v1 * (E1 + p1))
                else:
                    fluxes_rho_xi = ONE_TWO * (rhow0 + rhow1)
                    fluxes_u_xi = ONE_FOUR * (rhow0 + rhow1) * (u0 + u1)
                    fluxes_v_xi = ONE_FOUR * (rhow0 + rhow1) * (v0 + v1)
                    fluxes_w_xi = ONE_FOUR * (rhow0 + rhow1) * (w0 + w1) + p_mean
                    fluxes_E_xi = ONE_TWO * (w0 * (E0 + p0) + w1 * (E1 + p1))

            elif split_form == "BLAISDELL":

                p_mean = ONE_TWO * (p0 + p1)

                if axis == 0:
                    fluxes_rho_xi = ONE_FOUR * (rho0 + rho1) * (u0 + u1)
                    fluxes_u_xi = ONE_FOUR * (rhou0 + rhou1) * (u0 + u1) + p_mean
                    fluxes_v_xi = ONE_FOUR * (rhov0 + rhov1) * (u0 + u1)
                    fluxes_w_xi = ONE_FOUR * (rhow0 + rhow1) * (u0 + u1)
                    fluxes_E_xi = ONE_FOUR * (E0 + E1) * (u0 + u1) + ONE_TWO * (p0 * u0 + p1 * u1)

                elif axis == 1:
                    fluxes_rho_xi = ONE_FOUR * (rho0 + rho1) * (v0 + v1)
                    fluxes_u_xi = ONE_FOUR * (rhou0 + rhou1) * (v0 + v1) 
                    fluxes_v_xi = ONE_FOUR * (rhov0 + rhov1) * (v0 + v1) + p_mean
                    fluxes_w_xi = ONE_FOUR * (rhow0 + rhow1) * (v0 + v1)
                    fluxes_E_xi = ONE_FOUR * (E0 + E1) * (v0 + v1) + ONE_TWO * (p0 * v0 + p1 * v1)

                else:
                    fluxes_rho_xi = ONE_FOUR * (rho0 + rho1) * (w0 + w1)
                    fluxes_u_xi = ONE_FOUR * (rhou0 + rhou1) * (w0 + w1)
                    fluxes_v_xi = ONE_FOUR * (rhov0 + rhov1) * (w0 + w1)
                    fluxes_w_xi = ONE_FOUR * (rhow0 + rhow1) * (w0 + w1) + p_mean
                    fluxes_E_xi = ONE_FOUR * (E0 + E1) * (w0 + w1) + ONE_TWO * (p0 * w0 + p1 * w1) 

            elif split_form == "KENNEDY":
                
                e = E / rho
                e0 = e[s0]
                e1 = e[s1]

                p_mean = ONE_TWO * (p0 + p1)

                if axis == 0:
                    fluxes_rho_xi = ONE_FOUR * (rho0 + rho1) * (u0 + u1)
                    fluxes_u_xi = ONE_EIGHT * (rho0 + rho1) * (u0 + u1) * (u0 + u1) + p_mean
                    fluxes_v_xi = ONE_EIGHT * (rho0 + rho1) * (v0 + v1) * (u0 + u1)
                    fluxes_w_xi = ONE_EIGHT * (rho0 + rho1) * (w0 + w1) * (u0 + u1)
                    fluxes_E_xi = ONE_EIGHT * (rho0 + rho1) * (e0 + e1) * (u0 + u1) + ONE_TWO * (p0*u0 + p1*u1) 
                    # fluxes_E_xi = ONE_EIGHT * (rho0 + rho1) * (e0 + e1) * (u0 + u1) + ONE_FOUR * (p0 + p1) * (u0 + u1)

                elif axis == 1:
                    fluxes_rho_xi = ONE_FOUR * (rho0 + rho1) * (v0 + v1)
                    fluxes_u_xi = ONE_EIGHT * (rho0 + rho1) * (u0 + u1) * (v0 + v1) 
                    fluxes_v_xi = ONE_EIGHT * (rho0 + rho1) * (v0 + v1) * (v0 + v1) + p_mean
                    fluxes_w_xi = ONE_EIGHT * (rho0 + rho1) * (w0 + w1) * (v0 + v1)
                    fluxes_E_xi = ONE_EIGHT * (rho0 + rho1) * (e0 + e1) * (v0 + v1) + ONE_TWO * (p0*v0 + p1*v1) 
                    # fluxes_E_xi = ONE_EIGHT * (rho0 + rho1) * (e0 + e1) * (v0 + v1) + ONE_FOUR * (p0 + p1) * (v0 + v1)

                else:
                    fluxes_rho_xi = ONE_FOUR * (rho0 + rho1) * (w0 + w1)
                    fluxes_u_xi = ONE_EIGHT * (rho0 + rho1) * (u0 + u1) * (w0 + w1)
                    fluxes_v_xi = ONE_EIGHT * (rho0 + rho1) * (v0 + v1) * (w0 + w1)
                    fluxes_w_xi = ONE_EIGHT * (rho0 + rho1) * (w0 + w1) * (w0 + w1) + p_mean
                    fluxes_E_xi = ONE_EIGHT * (rho0 + rho1) * (e0 + e1) * (w0 + w1) + ONE_TWO * (p0*w0 + p1*w1)
                    # fluxes_E_xi = ONE_EIGHT * (rho0 + rho1) * (e0 + e1) * (w0 + w1) + ONE_FOUR * (p0 + p1) * (w0 + w1) 
            
            elif split_form.startswith("KEEP"):

                rho_mean = ONE_TWO * (rho0 + rho1)
                u_mean = ONE_TWO * (u0 + u1)
                v_mean = ONE_TWO * (v0 + v1)
                w_mean = ONE_TWO * (w0 + w1)
                p_mean = ONE_TWO * (p0 + p1)

                if axis == 0:
                    fluxes_rho_xi = rho_mean * u_mean
                    fluxes_u_xi = rho_mean * u_mean * u_mean + p_mean
                    fluxes_v_xi = rho_mean * v_mean * u_mean
                    fluxes_w_xi = rho_mean * w_mean * u_mean

                elif axis == 1:
                    fluxes_rho_xi = rho_mean * v_mean
                    fluxes_u_xi = rho_mean * u_mean * v_mean
                    fluxes_v_xi = rho_mean * v_mean * v_mean + p_mean
                    fluxes_w_xi = rho_mean * w_mean * v_mean

                else:
                    fluxes_rho_xi = rho_mean * w_mean
                    fluxes_u_xi = rho_mean * u_mean * w_mean
                    fluxes_v_xi = rho_mean * v_mean * w_mean
                    fluxes_w_xi = rho_mean * w_mean * w_mean + p_mean

                if split_form == "KEEP":
                    e0 = self.material_manager.get_specific_energy(p0, rho0)
                    e1 = self.material_manager.get_specific_energy(p1, rho1)

                    if axis == 0:
                        fluxes_E_xi = ONE_TWO * rho_mean * u_mean * (
                            (u0 * u1 + v0 * v1 + w0 * w1) + (e0 + e1)
                        ) + ONE_TWO * (u1 * p0 + u0 * p1)

                    elif axis == 1:
                        fluxes_E_xi = ONE_TWO * rho_mean * v_mean * (
                            (u0 * u1 + v0 * v1 + w0 * w1) + (e0 + e1)
                        ) + ONE_TWO * (v1 * p0 + v0 * p1)

                    else:
                        fluxes_E_xi = ONE_TWO * rho_mean * w_mean * (
                            (u0 * u1 + v0 * v1 + w0 * w1) + (e0 + e1)
                        ) + ONE_TWO * (w1 * p0 + w0 * p1)

                elif split_form == "KEEP-PE":
                    rhoe0 = self.material_manager.material.get_volumetric_energy(p0)
                    rhoe1 = self.material_manager.material.get_volumetric_energy(p1)

                    if axis == 0:
                        fluxes_E_xi = ONE_TWO * u_mean * (
                            rho_mean * (u0 * u1 + v0 * v1 + w0 * w1)
                            + (rhoe0 + rhoe1)
                        ) + ONE_TWO * (u1 * p0 + u0 * p1)

                    elif axis == 1:
                        fluxes_E_xi = ONE_TWO * v_mean * (
                            rho_mean * (u0 * u1 + v0 * v1 + w0 * w1)
                            + (rhoe0 + rhoe1)
                        ) + ONE_TWO * (v1 * p0 + v0 * p1)

                    else:
                        fluxes_E_xi = ONE_TWO * w_mean * (
                            rho_mean * (u0 * u1 + v0 * v1 + w0 * w1)
                            + (rhoe0 + rhoe1)
                        ) + ONE_TWO * (w1 * p0 + w0 * p1)

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            fluxes_xi = jnp.stack([
                fluxes_rho_xi,
                fluxes_u_xi,
                fluxes_v_xi,
                fluxes_w_xi,
                fluxes_E_xi,
            ])

        return fluxes_xi, None, None, None, None