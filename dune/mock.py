from dune.profiles import nfw
import numpy as np
import astropy.constants as c
import astropy.units as u

from .profiles import PlummerSampler
from .profiles import NFW, VelocitySampler


def GeneratePlummerNFW(
    N, rp, a, rho=(6.4e7 * u.M_sun / (u.kpc ** 3)), mass=500 * u.solMass, sphCoords=False, getProfile=False
):
    # Sample stars and initialize DM profile
    ssp = PlummerSampler()
    coords = ssp.generate_sph(N, rp)

    nfw_profile = NFW(a, rho, mass, rp)

    # Get kinematics
    sigma = nfw_profile.get_dispersion(coords["r"])
    vels = VelocitySampler(nfw_profile.nfw_potential, coords["r"], np.sqrt(sigma))

    if sphCoords == False:
        coords = ssp.convert_to_cartesian(coords)

    if getProfile:
        return ({**coords, **vels}, nfw_profile)

    return {**coords, **vels}
