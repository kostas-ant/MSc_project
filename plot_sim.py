#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.optimize import curve_fit
import sympy as sym


pos_i = np.loadtxt('pos/positions_a1_N10000')
pos_f = np.loadtxt('pos/positions_f_a1_N10000')
vel = np.loadtxt('vel/velocities_a01_N10000')
M_halo = np.load('M_halo_a1_N10000.npy')
t_end = np.load('t_end.npy')
m = np.loadtxt('masses/mass_p_f_a1_N10000')
# nDM = np.load('nDM.npy')
nDM = 3000
a = 1 # scale radius
rho0 = 2178.7925421492628 # scale density
def P(r, a=a, rho0=rho0):
    return 4*np.pi*r**2 * rho0 / ((r/a)*(1+r/a)**3)

def model(x, *par):
    return 4*np.pi* par[0] * x**2 / ((x**par[1])*(1+x)**par[2])

# radial positions of particles
r_i = (pos_i[:,0]**2 + pos_i[:,1]**2 + pos_i[:,2]**2)**0.5
r_f = (pos_f[:,0]**2 + pos_f[:,1]**2 + pos_f[:,2]**2)**0.5
print('position of BH: ', pos_f[-1,:])
print('r_BH = ', (pos_f[-1,0]**2 + pos_f[-1,1]**2 + pos_f[-1,2]**2)**0.5)
# calculate dynamical timescale
v_mean = np.mean((vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)) **0.5
R_mean = np.mean(r_i)
t_dyn = R_mean/v_mean * 3.171e-8 # convert to yr
print('dynamical timescale = %.2f Myr' %(t_dyn/1e6))


## plot histogram of radii
fig = plt.figure(figsize=(13,5))
nbins = 15 # number of bins

hist, bins1 = np.histogram((r_i), density=True, bins=nbins)
hist2, bins2 = np.histogram((r_f),bins=nbins, density=True)
bin_cent2 = bins2[:-1] + np.diff(bins2) / 2

# params, cov = curve_fit(model, bin_cent2, hist2, p0=[1e-2, 1.1, 2.1])
# print('params=', params)

r = np.geomspace(bins1[0], bins1[-1], 500)
r2 = np.geomspace(bins2[0], bins2[-1], 500)
norm = quad(P, bins1[0], bins1[-1])[0] # normalise P(r)

logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
logbins2 = np.logspace(np.log10(bins2[0]),np.log10(bins2[-1]),len(bins2))

# plot histogram of r
plt.subplot(121)
plt.title('N= %.f, a=%.2f pc, soft_length=1e-3 pc, t$_{dyn}$=%.2f Myr' %(nDM, a, (t_dyn/1e6)))
n1, bins1, _ = plt.hist((r_i), density=True, bins=logbins1, label='t=0')
n2, bins2, _ = plt.hist((r_f),bins=logbins2, density=True, alpha=0.5, label='t=%.3f t$_{dyn}$' %(t_end/t_dyn))
plt.plot((r), P(r)/norm, label='P(r)')
plt.yscale('log')
plt.xscale('log')
# plt.xlim(700,4900e3)
# plt.ylim(1e-9,1e-3)
plt.xlabel('r [pc]')
plt.legend()

# plot density profile
plt.subplot(122)
bin_cent1 = bins1[:-1] + np.diff(bins1) / 2
bin_cent2 = bins2[:-1] + np.diff(bins2) / 2
plt.plot(bin_cent1, norm * n1/(4*np.pi*bin_cent1**2), label='t=0')
plt.plot(bin_cent2, norm * n2/(4*np.pi*bin_cent2**2), label='t=%.e yr' %t_end)

# plot theoretical distribution
plt.plot((r), P(r)/(4*np.pi*r**2), label='theoretical initial distribution') # normalise integral to 1
# plt.plot(r2, model(r2,*params))
plt.yscale('log')
plt.xscale('log')
plt.ylabel('ρ [M$_\odot$/pc$^3$]')
plt.xlabel('r [pc]')
plt.legend()
plt.savefig('plots/dens_a1_soft1e-3_N10000')
plt.show()


## 3D plot of positions of particles
def plot3d(pos, title, color, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.scatter(pos[:,0]/1e3, pos[:,1]/1e3, pos[:,2]/1e3, zdir='z', color=color)
    # ax.set_xlim(-1500,1500)
    # ax.set_ylim(-1500,1500)
    # ax.set_zlim(-1500,1500)

    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    plt.savefig(path)

title1 = 'initial'
path1 = 'plots/BH_sim/halo3d_i'
title2 = 'after %.e yr' %(t_end)
path2 = 'plots/BH_sim/halo3d_f'
# plot3d(pos_i, title1,'C0', path1)
# plot3d(pos_f, title2,'C3', path2)


# calculate r_h
# def r_h(M_bh, a=a, rho0=rho0):
#     rh = sym.Symbol('$r_h$', positive=True)
#     return (sym.solve(2* np.pi*rho0 *(a**3* rh**2)/(2 *(a + rh)**2)-2*M_bh, rh))
#
# print('r_h=', r_h(2.6e6))
