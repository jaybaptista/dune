import numpy as np
import astropy.constants as c
import astropy.units as u

from .profiles import PlummerSampler
from .profiles import NFW, Hernquist, VelocitySampler

def GeneratePlummerNFW(N, rp, a, rho, mass):
  # Sample stars and initialize DM profile
  ssp = PlummerSampler()
  coords      = ssp.generate_sph(N, rp)
  nfw_profile = NFW(a, rho, mass, rp)

  # Get kinematics
  sigma = nfw_profile.get_dispersion(coords['r'])
  vels  = VelocitySampler(nfw_profile.nfw_potential, coords['r'], np.sqrt(sigma))

  return {**coords, **vels}