import numpy as np
import astropy.units as u

class PlummerSampler():
  
  def __init__(self):
    pass
  
  def plummer_density(self, r, dyn_mass, rp):
      factor = (3*dyn_mass) / (4 * np.pi * rp**3)
      ratio = r/rp
      return factor * (1 + (ratio**2))**(-5/2)

  def generate_radii(self, N, a):
    '''
    A function that generates random radial distances given a Plummer radius and number of stars.

    Parameters
    ----------
    N: int
      number of radii to generate
    a: double
      Plummer radius (in kpc)

    Returns
    -------
    np.array
      an array of radii generated using the Plummer inverse transform function
    '''
    mass = np.random.uniform(size=N)
    return ((mass**(-2/3) - 1) / (a**2))**(-1/2)
  
  def generate_theta(self, N):
    '''
    A function that generates random azimuthal angles uniformly between 0 and 2pi.

    Parameters
    ----------
    N: int
      number of angles to generate
    
    Returns
    -------
    np.array
      an array of sampled azimuthal angles
    '''
    return np.random.uniform(0, 2*np.pi, N)
  
  def generate_phi(self, N):
    '''
    A function that generates random polar angles.

    Parameters
    ----------
    N: int
      number of angles to generate
    
    Returns
    -------
    np.array
      an array of sampled polar angles
    '''
    return np.arccos(1-(2*np.random.uniform(0,1,N)))

  def generate_sph(self, N, a):
    '''
    A function that generates polar coordinates for randomly sampled points according to a Plummer profile.

    Parameters
    ----------
    N: int
      number of points to generate
    a: double
      Plummer radius (in kpc)
    
    Returns
    -------
    dict
      spherical coordinates for the generated points
    '''
    return {
      'r': self.generate_radii(N, a), 'theta': self.generate_theta(N), 'phi':self.generate_phi(N)
    }

  def convert_to_cartesian(self, coords):
    '''
    A function that converts spherical coordinates into Cartesian coordinates.

    Parameters
    ----------
    coords: dict
      A dictionary of sampled positions in spherical coordinates
    
    Returns
    -------
    dict
      Cartesian coordinates for the generated points
    '''

    r = coords['r']
    phi = coords['phi']
    theta = coords['theta']

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return {
      'x': x,
      'y': y,
      'z': z
    }

  def generate_cartesian(self, N, a):
    '''
    A function that generates Cartesian coordinates for randomly sampled points according to a Plummer profile.

    Parameters
    ----------
    N: int
      number of points to generate
    a: double
      Plummer radius (in kpc)
    
    Returns
    -------
    dict
      Cartesian coordinates for the generated points
    '''

    coords = self.generate_sph(N, a)
    return self.convert_to_cartesian(coords)