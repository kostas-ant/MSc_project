#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.optimize import curve_fit

from matplotlib import rc
plt.rcParams.update({'font.size': 26})
rc('text', usetex=True)
rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern']})

nDM = 100000
a = 100 # scale radius
rho0 = 2.178792542149263 # scale density for a=100
# rho0 = 0.21787925421492627
M_halo = 2*np.pi*rho0*a**3 # np.load('M_halo_a100.npy')


pos_i = np.loadtxt('pos/positions_a100_N100000')
pos_f = np.loadtxt('pos/positions_f_a100_N30000bh')
vel = np.loadtxt('vel/velocities_a100_N100000')
t_end = 1.1e8 #np.load('t_end.npy')
# m = np.loadtxt('masses/mass_p_f_a100_N4000bh_ph4')
# nDM = np.load('nDM.npy')

def P(r, a=a, rho0=rho0):
    return 4*np.pi*r**2 * rho0 / ((r/a)*(1+r/a)**3)

def model(x, *par):
    return 4*np.pi* par[0] * x**2 / ((x**par[1])*(1+x)**par[2])

# radial positions of particles
def r_pos(pos):
    return (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5

r_i = r_pos(pos_i)
r_f = r_pos(pos_f)
print('position of BH: ', pos_f[-1,:])
print('r_BH = ', (pos_f[-1,0]**2 + pos_f[-1,1]**2 + pos_f[-1,2]**2)**0.5)

# calculate dynamical timescale
r_h = 118
v_mean = np.mean((vel[:,0][(r_i<=r_h) & (r_i>0.9*r_h)]**2 + vel[:,1][(r_i<=r_h) & (r_i>0.9*r_h)]**2 + vel[:,2][(r_i<=r_h) & (r_i>0.9*r_h)]**2)) **0.5
R_mean = np.mean(r_i[(r_i<=r_h) & (r_i>0.1*r_h)])

t_dyn = r_h/v_mean * 3.171e-8 # convert to yr
print('dynamical timescale = %.4f Myr' %(t_dyn/1e6))


##  histogram of radii
nbins = 50 # number of bins

hist, bins1 = np.histogram((r_i), density=True, bins=nbins)
hist2, bins2 = np.histogram((r_f),bins=nbins, density=True)

# params, cov = curve_fit(model, bin_cent2, hist2, p0=[1e-2, 1.1, 2.1])
# print('params=', params)

r = np.geomspace(bins1[0], bins1[-1], 500)
r2 = np.geomspace(bins2[0], bins2[-1], 500)
norm = quad(P, bins1[0], bins1[-1])[0] # normalise P(r)

logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
logbins2 = np.logspace(np.log10(bins2[0]),np.log10(bins2[-1]),len(bins2))



# plot histogram of r
fig = plt.figure(figsize=(14,6))
plt.subplots_adjust(left=0.08, right=0.95, wspace=0.23, top=0.9)
fsize=22

plt.subplot(121)
plt.suptitle(r'N= $10^5$, a=%.f pc, soft. length=1 pc, $t_{dyn}(r \leq r_h)=%.1f$ Myr' %( a, (t_dyn/1e6)))
n1, bins1, _ = plt.hist((r_i), density=True, bins=logbins1, label='t=0')
n2, bins2, _ = plt.hist((r_f),bins=logbins2, density=True, alpha=0.5, label='t=%.2f t$_{dyn}$' %(t_end/t_dyn))
plt.plot((r), P(r)/norm, label='P(r)')

plt.yscale('log')
plt.xscale('log')
# plt.xlim(700,4900e3)
# plt.ylim(1e-9,1e-3)
plt.xlabel('r [pc]')
plt.legend(fontsize=fsize)

# plot density profile
plt.subplot(122)
bin_cent1 = 10**((np.log10(bins1[:-1]) + np.log10(bins1[1:])) /2) # + np.diff(bins1) / 2
bin_cent2 = 10**((np.log10(bins2[:-1]) + np.log10(bins2[1:])) /2) # + np.diff((bins2)) / 2
# plt.plot(bin_cent1, norm * n1/(4*np.pi*bin_cent1**2), label='t=0')
func = lambda x: n2[0]

# calculate density error in each bin
yerr = np.zeros(len(n2))
for i in range(len(bins2[:-1])):
    ri = r_f[(r_f>=bins2[i]) & (r_f<bins2[i+1])]
    rhoi = norm * n2[i]/(4*np.pi*ri**2)
    yerr[i] = np.sqrt(len(ri)) /len(ri) *n2[i]* norm/(4*np.pi*bin_cent2[i]**2) #*(bins2[i+1]-bins2[i])


rho_f = norm * n2/(4*np.pi*bin_cent2**2)
# np.savetxt('rho_sim_100000_soft1_a100_m2e6long', np.array([rho_f, yerr]).T, header='rho, rho_err')
# np.savetxt('rbin_100000_soft1_a100_m2e6long', np.array([bin_cent2,bin_cent2-bins2[:-1], bins2[1:]-bin_cent2]).T, header='bin_cent, xerr-, xerr+')
#

slope = (np.log10(rho_f[8]) - np.log10(rho_f[11])) / (np.log10(bin_cent2[8]) - np.log10(bin_cent2[11]))
print('spike slope =', slope)

# plot theoretical distribution
plt.plot((r), P(r)/(4*np.pi*r**2), label='Initial density profile') # normalise integral to 1
plt.errorbar(bin_cent2, rho_f, xerr=[bin_cent2-bins2[:-1], bins2[1:]-bin_cent2], yerr=yerr, fmt='.' , label='t=%.f Myr' %(t_end/1e6))

# plt.plot(r2, model(r2,*params))
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$\rho$ [$M_\odot/$pc$^3$]')
plt.xlabel('r [pc]')
plt.legend(fontsize=22)
# plt.savefig('plots/BH/dens_a100_soft1_N100000bh.pdf', bbox_inches='tight', format='pdf')
# plt.show()


## plot densities for different N
def densN(posf, N):

    # posf = np.loadtxt(pathN)
    # posf =

    rf = r_pos(posf)

    histf, binsf = np.histogram((rf),bins=35, density=True)

    rf2 = np.geomspace(binsf[0], binsf[-1], 500)
    logbinsf = np.logspace(np.log10(binsf[0]),np.log10(binsf[-1]),len(binsf))
    nf, binsf, = np.histogram(rf, bins=logbinsf, density=True)
    bin_centf =  10**((np.log10(binsf[:-1]) + np.log10(binsf[1:])) /2)

    yerr = np.zeros(len(nf))
    for i in range(len(binsf[:-1])):
        ri = r_f[(r_f>=binsf[i]) & (r_f<binsf[i+1])]
        rhoi = norm * nf[i]/(4*np.pi*ri**2)
        yerr[i] = np.sqrt(len(ri)) /len(ri) *nf[i]* norm/(4*np.pi*bin_centf[i]**2)
    rho_f = norm * nf/(4*np.pi*bin_centf**2)


    plt.errorbar(bin_centf, rho_f, xerr=[bin_centf-binsf[:-1], binsf[1:]-bin_centf], yerr=yerr, fmt='.' , label=r'$t$ = %.f Myr' %N)

    # plt.plot(bin_centf, norm * nf/(4*np.pi*bin_centf**2), label=r'$m_{BH}$ = %.f$\times10^6M_\odot$' %N)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\rho$ [M$_\odot$/pc$^3$]')
    plt.xlabel('r [pc]')
    plt.legend(fontsize=22)


## 3D plot of positions of particles
def plot3d(pos, title, color, path):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.scatter(pos[:,0]/1e3, pos[:,1]/1e3, pos[:,2]/1e3, zdir='z', color=color, s=12)
    ax.set_xlim(-0.100,0.100)
    ax.set_ylim(-0.100,0.100)
    ax.set_zlim(-0.100,0.100)
    # ax.tick_params(axis='x', labelsize=20)
    # ax.tick_params(axis='y', labelsize=20)
    # ax.tick_params(axis='z', labelsize=20)

    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    plt.savefig(path)

# title1 = 'initial'
# path1 = 'plots/BH/halo3d_i'
# title2 = r'after %.1f $\times 10^7$ yr' %(t_end/1e7)
# path2 = 'plots/BH/halo3d_f'
# plot3d(pos_i[::15,:], title1,'C0', path1)
# plot3d(pos_f[::15,:], title2,'C3', path2)

plt.figure(figsize=(10,8))
# # plt.title('a=%.f pc, soft_length=1e-5 pc, t$_{dyn}$=%.2f Myr, t$_f$=%.2f t$_{dyn}$' %(a, (t_dyn/1e6), t_end/t_dyn))
plt.plot((r), P(r)/(4*np.pi*r**2), label='Theor. init. distribution', linestyle='--', color='k')
#
# densN('pos/positions_f_a100_N100000bh', 71)
px = np.loadtxt('pos/step/positions_f_a100_N100000stepx')
py = np.loadtxt('pos/step/positions_f_a100_N100000stepy')
pz = np.loadtxt('pos/step/positions_f_a100_N100000stepz')
pf1 = np.array([px[20,:], py[20,:], pz[20,:]]).T
pf2 = np.array([px[50,:], py[50,:], pz[50,:]]).T
pf3 = np.array([px[99,:], py[99,:], pz[99,:]]).T
densN(pf1, 0.2*65)
densN(pf2, 0.5*65)
densN(pf3, 65)
bh1 = ((pf1[-1,0]**2 + pf1[-1,1]**2 + pf1[-1,2]**2)**0.5)
bh2 = ((pf2[-1,0]**2 + pf2[-1,1]**2 + pf2[-1,2]**2)**0.5)
bh3 = ((pf3[-1,0]**2 + pf3[-1,1]**2 + pf3[-1,2]**2)**0.5)
# plt.errorbar(bin_cent2, rho_f, xerr=[bin_cent2-bins2[:-1], bins2[1:]-bin_cent2], yerr=yerr, fmt='.' , label='t=%.f Myr' %(t_end/1e6))
# print(pf1.shape, px.shape)
#
plt.axvline(bh1, linestyle=':')
# densN('pos/positions_f_a100_N100000_bh2_100_long', 112)
plt.axvline(bh2, linestyle=':', color='C1')
# densN('pos/positions_f_a100_N100000_bh2_100_longer', 223)
plt.axvline(bh3, linestyle=':', color='C2')
plt.ylim(1e-1, 1e7)
plt.xlim(1e-2, 200)
#
#
# #
# plt.text(2,0.02, '$r_{BH}$', rotation=-90)
# rho_H = (M_halo/(2*np.pi*a**3)) / ((bin_cent2/a)*(1+bin_cent2/a)**3)
# plt.plot(bin_cent2, rho_H, linestyle='--', color='k')
plt.savefig('plots/BH/dens_timesteps2', bbox_inches='tight', format='pdf')
#
plt.show()

# densN('pos/positions_f_a1_N3000', 3000)
# plt.show()
# calculate r_h
# import sympy as sym
# def r_h(M_bh, a=a, rho0=rho0):
#     rh = sym.Symbol('$r_h$', positive=True)
#     return (sym.solve(2* np.pi*rho0 *(a**3* rh**2)/(2 *(a + rh)**2)-2*M_bh, rh))
#
# print('r_h=', r_h(2.6e6))
