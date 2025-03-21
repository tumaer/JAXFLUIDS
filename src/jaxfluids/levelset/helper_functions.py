import jax
import jax.numpy as jnp

from jaxfluids.domain.domain_information import DomainInformation

Array = jax.Array

def transform_to_conserved(
        conservatives: Array,
        volume_fraction: Array,
        domain_information: DomainInformation,
        levelset_model: str
        ) -> Array:
    nhx, nhy, nhz = domain_information.domain_slices_conservatives
    nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
    if levelset_model == "FLUID-FLUID":
        volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
    conservatives = conservatives.at[...,nhx,nhy,nhz].mul(volume_fraction[...,nhx_,nhy_,nhz_])
    return conservatives

def transform_to_volume_average(
        conservatives: Array,
        volume_fraction: Array,
        domain_information: DomainInformation
        ) -> Array:
    nhx, nhy, nhz = domain_information.domain_slices_conservatives
    nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
    mask = volume_fraction[...,nhx_,nhy_,nhz_] == 0.0
    denominator = volume_fraction[...,nhx_,nhy_,nhz_] + mask * 1e-20
    conservatives = conservatives.at[...,nhx,nhy,nhz].mul(1.0/denominator)
    return conservatives

def weight_cell_face_flux_xi(
        flux_xi: Array,
        apertures: Array,
        domain_information: DomainInformation,
        levelset_model: str
        ) -> Array:
    """Weights the flux at a given cell face
    with the corresponding aperture. 

    :param flux_xi: Cell-face flux in xi-direction
    :type flux_xi: Array
    :param apertures: Aperture in xi-direction
    :type apertures: Array
    :param domain_information: DomainInformation
    :type domain_information: DomainInformation
    :param levelset_model: String identifier of the active level-set model
    :type levelset_model: str
    :return: Aperture-weighted flux
    :rtype: Array
    """
    nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
    if levelset_model == "FLUID-FLUID": 
        apertures = jnp.stack([apertures, 1.0 - apertures], axis=0)
    flux_xi *= apertures[...,nhx_,nhy_,nhz_]
    return flux_xi

def weight_volume_force(
        volume_force: Array,
        volume_fraction: Array,
        domain_information: DomainInformation,
        levelset_model: str
        ) -> Array:
    nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
    if levelset_model == "FLUID-FLUID":
        volume_fraction = jnp.stack([volume_fraction, 1.0 - volume_fraction], axis=0)
    volume_force *= volume_fraction[...,nhx_,nhy_,nhz_]
    return volume_force

def weight_solid_energy(
        solid_energy: Array,
        volume_fraction: Array,
        domain_information: DomainInformation,
        ) -> Array:
    nhx, nhy, nhz = domain_information.domain_slices_conservatives
    nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
    volume_fraction_solid = 1.0 - volume_fraction[...,nhx_,nhy_,nhz_]
    solid_energy = solid_energy.at[...,nhx,nhy,nhz].mul(volume_fraction_solid)
    return solid_energy

