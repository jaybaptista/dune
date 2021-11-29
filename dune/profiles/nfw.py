from .plummer import PlummerSampler
from scipy.integrate import quad
import numpy as np
import astropy.constants as c
import astropy.units as u
from scipy.stats import truncnorm


class NFW():

    def __init__(self, a=(.01 * u.kpc), rho=(6.4e7 * u.M_sun / (u.kpc**3)), dyn_mass=500*u.solMass, rp=.2*u.kpc):
        '''
        Creates a NFW profile with given parameters

        Parameters
        ----------
        a: astropy.units.Quantity
            scale radius
        rho: astropy.units.Quantity
            dark matter density
        dyn_mass: astropy.units.Quantity
            bound halo mass
        rp: astropy.units.Quantity
            Plummer radius
        '''
        self.a = a.decompose()
        self.rho = rho.decompose()
        self.mass = dyn_mass.decompose()
        self.rp = rp.decompose()

    def nfw_potential(self, r, a=None, rho=None):
        '''
        Evaluates the potential at a given radius
        Parameters
        ----------
        r: astropy.units.Quantity
            radius to evaluate potential
        a: astropy.units.Quantity (default NFW.a)
            scale radius
        rho: astropy.units.Quantity (default NFW.rho)
            dark matter density

        Returns
        -------
        astropy.units.Quantity
            NFW potential at radius r
        '''

        if a is None:
            a = self.a
        else:
            a = a.decompose()

        if rho is None:
            rho = self.rho
        else:
            rho = rho.decompose()

        if type(r) != u.quantity.Quantity:
            raise ValueError('radius must be specified in astropy units')

        if type(a) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')

        if (type(rho) != u.quantity.Quantity):
            raise ValueError('value must be specified in astropy units')

        factor = -4*np.pi*c.G*rho*(a**2)
        ratio = r/a

        return factor * np.log(1 + ratio) / ratio

    def nfw_force(self, r, a=None, rho=None):
        '''
        Evaluates the gravitational force at a given radius
        Parameters
        ----------
        r: astropy.units.Quantity
            radius to evaluate potential
        a: astropy.units.Quantity (default NFW.a)
            scale radius
        rho: astropy.units.Quantity (default NFW.rho)
            dark matter density

        Returns
        -------
        astropy.units.Quantity
            NFW force at radius r
        '''
        if a is None:
            a = self.a
        else:
            a = a.decompose()

        if rho is None:
            rho = self.rho
        else:
            rho = rho.decompose()

        if type(r) != u.quantity.Quantity:
            raise ValueError('radius must be specified in astropy units')

        if type(a) != u.quantity.Quantity:
            raise ValueError('value must be specified in astropy units')

        if (type(rho) != u.quantity.Quantity):
            raise ValueError('value must be specified in astropy units')

        factor = 4*np.pi*c.G*rho*(a**2)
        ratio = r/a
        return factor * ((1/(r*(ratio+1))) - ((1/(ratio * r)) * np.log(1 + ratio)))

    def get_dispersion(self, r, a=None, rho=None, dyn_mass=None, rp=None):
        '''
        Evaluates the potential at a given radius
        Parameters
        ----------
        r: astropy.units.Quantity
            radius to evaluate potential
        a: astropy.units.Quantity (default NFW.a)
            scale radius
        rho: astropy.units.Quantity (default NFW.rho)
            dark matter density
        dyn_mass: astropy.units.Quantity (default NFW.mass)
            bound halo mass
        rp: astropy.units.Quantity (default NFW.rp)
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

        if rho is None:
            rho = self.rho
        else:
            rho = rho.decompose()

        if rp is None:
            rp = self.rp
        else:
            rp = rp.decompose()

        if dyn_mass is None:
            dyn_mass = self.mass
        else:
            dyn_mass = dyn_mass.decompose()

        # integrated_units = a.unit * rho.unit * \
        #     (u.m**3) * (u.s**(-2)) * (1/u.kg) * \
        #     dyn_mass.unit * (rp.unit**(-3)) * r.unit

        integrated_units = (a.unit * rho.unit * c.G.unit * dyn_mass.unit * rp.unit**(-3) * r.unit).decompose()

        r_unit = r.unit

        tmp_ssp = PlummerSampler()

        def integrand(r, dyn_mass, rp, a, rho):
            r = r * r_unit

            return -1*((self.nfw_force(r, a, rho).decompose()).value)*((tmp_ssp.plummer_density(r, dyn_mass, rp).decompose()).value)

        if r.size > 1:
            _sum = [quad(integrand, r_i, np.inf, args=(
                dyn_mass, rp, a, rho))[0] for r_i in r.value]
            output = ((1/(tmp_ssp.plummer_density(r, dyn_mass, rp)).decompose())
                      * _sum * integrated_units)
            return output
        else:
            output = ((1/(tmp_ssp.plummer_density(r, dyn_mass, rp)).decompose()) * quad(integrand,
                      r.value, np.inf, args=(dyn_mass, rp, a, rho)) * integrated_units)
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
                (2*abs(self.nfw_potential(r[i])))**(1/2)).to(u.km / u.s).value

            a, b = -1*v_esc/sigma[i], v_esc/sigma[i]

            # Sample from a truncated normal distribution s.t. we ignore energies that launch star out of system
            vr = truncnorm.rvs(a, b, scale=sigma[i], loc=0) * u.km / u.s

            # Using whatever energy we have left from the prior drawn sample, select a valid tangential velocity
            max_vt = ((2*abs(self.nfw_potential(r[i])) -
                       (vr**2))**(1/2)).to(u.km / u.s).value
            vt = truncnorm.rvs(-1*max_vt/sigma[i], max_vt /
                               sigma[i], scale=sigma[i], loc=0) * u.km / u.s

            if (not with_units):
                vr = vr.value
                vt = vt.value

            v['vr'].append(vr)
            v['vt'].append(vt)

        return v
