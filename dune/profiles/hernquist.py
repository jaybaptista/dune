from .plummer import PlummerSampler
from scipy.integrate import quad
import numpy as np
import astropy.constants as c
import astropy.units as u
from scipy.stats import truncnorm


class Hernquist():

    def __init__(self, a=(.01 * u.kpc), dyn_mass=500*u.solMass):
        '''
        Creates a Hernquist profile with given parameters

        Parameters
        ----------
        a: astropy.units.Quantity
            scale radius
        dyn_mass: astropy.units.Quantity
            bound halo mass
        rp: astropy.units.Quantity
            Plummer radius
        '''
        self.a = a.decompose()
        self.mass = dyn_mass.decompose()


    def hq_density(self, r):
        return self.mass * self.a / (2*np.pi*r*(r+self.a)**(3))

    def generate_radii(self, N):
        '''
        A function that generates random radial distances given a scale radius and number of stars.

        Parameters
        ----------
        N: int
            number of radii to generate
        a: double
            scale radius (in kpc)

        Returns
        -------
        np.array
            an array of radii generated using the Plummer inverse transform function
        '''

        mass = np.random.uniform(size=N)
        return ((mass**(-1/2) - 1) / (self.a))**(-1)

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

    def generate_sph(self, N):
        '''
        A function that generates polar coordinates for randomly sampled points according to a Hernquist profile.

        Parameters
        ----------
        N: int
            number of points to generate
        
        Returns
        -------
        dict
            spherical coordinates for the generated points
        '''
        return {
        'r': self.generate_radii(N), 'theta': self.generate_theta(N), 'phi':self.generate_phi(N)
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

    def generate_cartesian(self, N):
        '''
        A function that generates Cartesian coordinates for randomly sampled points according to a Plummer profile.

        Parameters
        ----------
        N: int
            number of points to generate
        
        Returns
        -------
        dict
            Cartesian coordinates for the generated points
        '''

        coords = self.generate_sph(N, self.a)
        return self.convert_to_cartesian(coords)

    def enclosed_mass(self, r):
        return self.mass * (r**2) / (r + self.a)**2

    def hernquist_potential(self, r, a=None):
        '''
        Evaluates the potential at a given radius
        Parameters
        ----------
        r: astropy.units.Quantity
            radius to evaluate potential
        a: astropy.units.Quantity (default Hernquist.a)
            scale radius

        Returns
        -------
        astropy.units.Quantity
            Hernquist potential at radius r
        '''

        if a is None:
            a = self.a
        else:
            a = a.decompose()

        if type(r) != u.quantity.Quantity:
            raise ValueError('radius must be specified in astropy units')

        if type(a) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')


        pot = -1*c.G * self.enclosed_mass(r) / (r+a)

        return pot

    def hernquist_force(self, r, a=None):
        '''
        Evaluates the gravitational force at a given radius
        Parameters
        ----------
        r: astropy.units.Quantity
            radius to evaluate potential
        a: astropy.units.Quantity (default Hernquist.a)
            scale radius

        Returns
        -------
        force: astropy.units.Quantity
            Hernquist force at radius r
        '''
        if a is None:
            a = self.a
        else:
            a = a.decompose()


        if type(r) != u.quantity.Quantity:
            raise ValueError('radius must be specified in astropy units')

        if type(a) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')

        force = -1*c.G * self.mass * r * (r-(2*a)) / (r+a)**4

        return force

    def get_dispersion(self, r, a=None, dyn_mass=None):
        '''
        Evaluates the potential at a given radius
        Parameters
        ----------
        r: astropy.units.Quantity
            radius to evaluate potential
        a: astropy.units.Quantity (default Hernquist.a)
            scale radius
        dyn_mass: astropy.units.Quantity (default Hernquist.mass)
            bound halo mass

        Returns
        -------
        astropy.units.Quantity
            the velocity dispersion at a given radius
        '''
        if a is None:
            a = self.a
        else:
            a = a.decompose()

        if dyn_mass is None:
            dyn_mass = self.mass
        else:
            dyn_mass = dyn_mass.decompose()

        self.integrated_units = None

        r_unit = r.unit

        def integrand(r):
            r = r * r_unit
            hq_force = (self.hernquist_force(r).decompose())
            density = (self.hq_density(r).decompose())
            self.integrated_units = hq_force.unit * density.unit * r_unit
            return -1*(hq_force.value)*(density.value)

        if r.size > 1:
            _sum = [quad(integrand, r_i, np.inf)[0] for r_i in r.value]
            output = ((1/(self.hq_density(r).decompose()))
                      * _sum * self.integrated_units)
            return output
        else:
            output = ((1/(self.hq_density(r)).decompose()) * quad(integrand,
                      r.value, np.inf) * self.integrated_units)
            return output[0]