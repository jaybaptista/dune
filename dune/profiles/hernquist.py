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
        self.a = a.decompose()
        self.mass = dyn_mass.decompose()
        self.rp = rp.decompose()

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
        else:
            a = a.decompose()

        if dyn_mass is None:
            dyn_mass = self.mass
        else:
            dyn_mass = dyn_mass.decompose()

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
        else:
            a = a.decompose()

        if dyn_mass is None:
            dyn_mass = self.mass
        else:
            dyn_mass = dyn_mass.decompose()

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
        else:
            a = a.decompose()

        if dyn_mass is None:
            dyn_mass = self.mass
        else:
            dyn_mass = dyn_mass.decompose()

        if rp is None:
            rp = self.rp
        else:
            rp = rp.decompose()

        self.integrated_units = None

        r_unit = r.unit

        tmp_ssp = PlummerSampler()

        def integrand(r, dyn_mass, rp, a):
            r = r * r_unit
            hq_force = (self.hernquist_force(r, a, dyn_mass).decompose())
            plum_density = (tmp_ssp.plummer_density(r, dyn_mass, rp).decompose())

            self.integrated_units = hq_force.unit * plum_density.unit * r_unit

            return -1*(hq_force.value)*(plum_density.value)

        if r.size > 1:
            _sum = [quad(integrand, r_i, np.inf, args=(
                dyn_mass, rp, a))[0] for r_i in r.value]
            output = ((1/(tmp_ssp.plummer_density(r, dyn_mass, rp).decompose()))
                      * _sum * self.integrated_units)
            return output
        else:
            output = ((1/(tmp_ssp.plummer_density(r, dyn_mass, rp)).decompose()) * quad(integrand,
                      r.value, np.inf, args=(dyn_mass, rp, a)) * self.integrated_units)
            return output[0]