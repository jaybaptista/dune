import numpy as np
import astropy.constants as c
import astropy.units as u
from scipy.stats import truncnorm

def VelocitySampler(potential, r, sigma, with_units=False):
        '''
        Evaluates the potential at a given radius
        Parameters
        ----------
        potential:
          potential function of an initialized profile class
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
            'vt': [],
            'vx': [],
            'vy': [],
            'vz': [],
            'vd': sigma
        }

        for i in np.arange(r.size):

            v_esc = (
                (2*abs(potential(r[i])))**(1/2)).to(u.km / u.s).value

            a, b = -1*v_esc/sigma[i], v_esc/sigma[i]

            # Sample from a truncated normal distribution s.t. we ignore energies that launch star out of system
            print(v_esc, sigma[i])
            vr = truncnorm.rvs(a, b, scale=sigma[i], loc=0) * u.km / u.s

            # Using whatever energy we have left from the prior drawn sample, select a valid tangential velocity
            max_vt = ((2*abs(potential(r[i])) -
                       (vr**2))**(1/2)).to(u.km / u.s).value
            vt = truncnorm.rvs(-1*max_vt/sigma[i], max_vt /
                               sigma[i], scale=sigma[i], loc=0) * u.km / u.s

            if (not with_units):
                vr = vr.value
                vt = vt.value


            v['vr'].append(vr)
            v['vt'].append(vt)

            v_length = (vr**2 + vt**2)**(1/2)

            v_xyz = np.random.normal(size=3)
            v_xyz = (v_length * v_xyz) / np.linalg.norm(v_xyz, axis=0)

            v['vx'].append(v_xyz[0])
            v['vy'].append(v_xyz[1])
            v['vz'].append(v_xyz[2])

        return v