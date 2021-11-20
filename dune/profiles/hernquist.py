from .plummer import PlummerSampler
from scipy.integrate import quad
import numpy as np
import astropy.constants as c
import astropy.units as u
from scipy.stats import truncnorm

class Hernquist():

    def __init__(self, a=(.01 * u.kpc), dyn_mass=500*u.solMass, rp=.2*u.kpc):
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
        self.a = a
        self.mass = dyn_mass
        self.rp = rp

    def hernquist_potential(self, r, a=None, dyn_mass=None):
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
        
        if dyn_mass is None:
          dyn_mass = self.mass

        if type(r) != u.quantity.Quantity:
            raise ValueError('radius must be specified in astropy units')

        if type(a) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')

        if type(dyn_mass) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')

        pot = -1*c.G * dyn_mass / (r+a)

        return pot

    def hernquist_force(self, r, a=None, dyn_mass=None):
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

        if dyn_mass is None:
          dyn_mass = self.mass

        if type(r) != u.quantity.Quantity:
            raise ValueError('radius must be specified in astropy units')

        if type(a) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')

        if type(dyn_mass) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')

        force = -1*c.G * dyn_mass / (r+a)**2

        return force

    def get_dispersion(self, r, a=None, dyn_mass=None, rp=None):
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
        rp: astropy.units.Quantity (default Hernquist.rp)
            Plummer radius

        Returns
        -------
        astropy.units.Quantity
            the velocity dispersion at a given radius
        '''
        if a is None:
            a = self.a


        if rp is None:
            rp = self.rp

        if dyn_mass is None:
            dyn_mass = self.mass

        integrated_units = a.unit * \
            (u.m**3) * (u.s**(-2)) * (1/u.kg) * \
            dyn_mass.unit * (rp.unit**(-3)) * r.unit

        r_unit = r.unit

        def integrand(r, dyn_mass, rp, a):
            r = r * r_unit
            return -1*(self.hernquist_force(r, a, dyn_mass).value)*(PlummerSampler.plummer_density(r, dyn_mass, rp).value)

        if r.size > 1:
            _sum = [quad(integrand, r_i, np.inf, args=(
                dyn_mass, rp, a))[0] for r_i in r.value]
            output = ((1/PlummerSampler.plummer_density(r, dyn_mass, rp))
                      * _sum * integrated_units)
            return output
        else:
            output = ((1/PlummerSampler.plummer_density(r, dyn_mass, rp)) * quad(integrand,
                      r.value, np.inf, args=(dyn_mass, rp, a)) * integrated_units)
            return output[0]

    def sample_velocity(self, r, sigma, with_units=False):
        '''
        Evaluates the potential at a given radius
        Parameters
        ----------
        r: astropy.units.Quantity
            radius to evaluate potential
        sigma: astropy.units.Quantity
            velocity dispersion
        with_units: boolean (default False)
            convert to astropy quantity in units of km/s

        Returns
        -------
        dict
            sampled radial and tangential velocities
        '''
        
        # Convert units for consistency
        r = (r.to(u.kpc))
        sigma = (sigma.to(u.km / u.s)).value

        v = {
            'vr': [],
            'vt': []
        }

        for i in np.arange(r.size):

            v_esc = (
                (2*abs(self.hernquist_potential(r[i])))**(1/2)).to(u.km / u.s).value

            a, b = -1*v_esc/sigma[i], v_esc/sigma[i]

            # Sample from a truncated normal distribution s.t. we ignore energies that launch star out of system
            vr = truncnorm.rvs(a, b, scale=sigma[i], loc=0) * u.km / u.s

            # Using whatever energy we have left from the prior drawn sample, select a valid tangential velocity
            max_vt = ((2*abs(self.hernquist_potential(r[i])) -
                       (vr**2))**(1/2)).to(u.km / u.s).value
            vt = truncnorm.rvs(-1*max_vt/sigma[i], max_vt /
                               sigma[i], scale=sigma[i], loc=0) * u.km / u.s

            if (not with_units):
                vr = vr.value
                vt = vt.value

            v['vr'].append(vr)
            v['vt'].append(vt)

        return v
