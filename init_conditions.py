import numpy as np
import astropy.units as units
import astropy.constants as const
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.integrate import quad

from tqdm import tqdm

import time

start = time.time()

# constants
c = const.c.to(units.parsec/units.s)
G = const.G.to(units.parsec**3/units.solMass/units.s**2)
# a = 27355.3624059166 *units.parsec # scale radius
# rho0 = 0.00796477308477558 *units.solMass/units.parsec**3
a = 1 *units.parsec # scale radius
rho0 = 217.87925421492628 *units.solMass/units.parsec**3
M_halo = 2*np.pi*rho0*a**3
M = M_halo

nDM = 10000 # number of particles
r = np.logspace(-8, np.log10(100), 1000)

def rho_DM(r, a=a.value, rho0=rho0.value):
    return rho0 / ((r/a)*(1+r/a)**3)
# Hernquist distribution function
def f_dist(eps):
    return eps**0.5/(1-eps)**2 * ((1-2*eps)*(8*eps**2-8*eps-3) + 3*np.arcsin(eps**0.5)/np.sqrt(eps*(1-eps)))

# constant coefficients of distribution function
def f_coef(M):
    return M / (2**0.5 * (2*np.pi)**3 * (G*M*a)**(3/2))

def psi_rel(r):
    return (G*M/(a+r*units.parsec)).value # Relative potential

def vmax(r):
    return np.sqrt(2*psi_rel(r))

# velocity distribution f(v) at a given radius r
@np.vectorize
def f(r, v, a=a.value, G=G.value, M=M):
    if (v >= vmax(r)):
        return 0.0
    else:
        return f_coef(M).value * 4*np.pi*v**2 * f_dist( a/(G*M.value)*(psi_rel(r) - 0.5*v**2) ) / rho_DM(r)


Menc = 4*np.pi*rho0 * a.value**3 * r**2 / (2*(a.value+r)**2) # CDF which is the same as the enclosed mass
Mmax = Menc[-1]
f_interp = interp1d(Menc/Mmax, r)
# particle positions
rvals = f_interp(np.random.rand(nDM))

# randomly generate spherical coordinates
ctvals = 2.0*np.random.rand(nDM) - 1.0
thetavals = np.arccos(ctvals)
phivals = 2*np.pi*np.random.rand(nDM)

# conversion to Cartesian
xvals = rvals*np.cos(phivals)*np.sin(thetavals)
yvals = rvals*np.sin(phivals)*np.sin(thetavals)
zvals = rvals*np.cos(thetavals)

# find maximum for Rejection sampling
# r = np.logspace(-6, -4, 5)
# print(vmax(1e-2))
# for ri in r:
#     v = np.geomspace(1e-10*vmax(ri),vmax(ri), 100)
#     plt.plot(v,f(ri,v), label=('r=%.1e' %ri))
#
#     plt.xlabel('v')
#     plt.ylabel('(f(v,r))')
#     plt.legend()
# plt.xlim(-1e-15, 1e-15)
# plt.savefig('test_f')
# plt.show()

# Generate velocities
vvals = np.zeros(nDM)
for ind in tqdm(range(nDM), desc="    Sampling velocities..."):
    count = 0
    r = rvals[ind]
    #Now sample f(v) at given r to get the speed v
    found = 0
    v_arr = np.geomspace(1e-4*vmax(r),vmax(r), 500)

    while (found == 0):
        count += 1
        if (count > 100000):
            print("Velocity sampling failed at r = ", r)
            sys.exit()

        #Rejection sampling for the velocities
        v = np.random.rand(1)*vmax(r)
        if ((np.random.rand(1)*1.01*max(f(r, v_arr))) < f(r, v)):
            found = 1
            vvals[ind] = v


# Get a new set of random directions for the velocities
ctvals2 = 2.0*np.random.rand(nDM) - 1.0
thetavals2 = np.arccos(ctvals2)
phivals2 = 2*np.pi*np.random.rand(nDM)

vxvals = vvals*np.cos(phivals2)*np.sin(thetavals2)
vyvals = vvals*np.sin(phivals2)*np.sin(thetavals2)
vzvals = vvals*np.cos(thetavals2)


# save arrays of positions and velocities
m_vals = np.zeros(nDM) # in Msun
pos_vals = np.zeros((nDM,3)) # in pc
vel_vals = np.zeros((nDM,3)) # in pc/s

# Set up BH
# m_vals[0] = 2.6e6
# pos_vals[0,:] = np.zeros(3)
# vel_vals[0,:] = np.zeros(3)

pos_vals[:,:] = np.array([xvals, yvals, zvals]).T
m_vals[:] = M_halo.value/nDM
header = 'a=%.f pc, r[1e-8,10], N=%.f' %(a.value, nDM)
np.savetxt('pos/positions_a1_N10000', pos_vals, header=header)
np.savetxt('masses/mass_particle_a1_N10000', m_vals, header=header)
np.save('M_halo_a1_N10000', M_halo.value)
np.save('nDM', nDM)

vel_vals[:,:] = np.array([vxvals, vyvals, vzvals]).T
np.savetxt('vel/velocities_a1_N10000', vel_vals, header=header)

end = time.time()
print('elapsed time = ', end-start)
