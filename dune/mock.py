from dune.profiles import nfw
import numpy as np
import astropy.constants as c
import astropy.units as u

from .profiles import PlummerSampler
from .profiles import NFW, Hernquist, VelocitySampler


def GeneratePlummerNFW(N, rp, a, rho=(6.4e7 * u.M_sun / (u.kpc**3)), mass=500*u.solMass):
    # Sample stars and initialize DM profile
    ssp = PlummerSampler()
    coords     = ssp.generate_sph(N, rp)
    coords_xyz = ssp.convert_to_cartesian(coords)
    nfw_profile = NFW(a, rho, mass, rp)

    # Get kinematics
    sigma = nfw_profile.get_dispersion(coords['r'])
    vels = VelocitySampler(nfw_profile.nfw_potential,
                           coords['r'], np.sqrt(sigma))

    return {**coords, **coords_xyz, **vels}, nfw_profile.nfw_potential, nfw_profile.nfw_force


def GenerateHernquist(N, a, mass):
    # Sample stars and initialize DM profile
    hq_profile = Hernquist(a, mass)
    coords     = hq_profile.generate_sph(N)
    coords_xyz = hq_profile.convert_to_cartesian(coords)

    # Get kinematics
    sigma = hq_profile.get_dispersion(coords['r'])
    vels = VelocitySampler(hq_profile.hernquist_potential,
                           coords['r'], np.sqrt(abs(sigma)))

    return {**coords, **coords_xyz, **vels}, hq_profile.hernquist_potential, hq_profile.hernquist_force