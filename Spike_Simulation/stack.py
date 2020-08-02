import numpy as np


N = 10 # number of files
nDM = 100000 # final total number of particles

pnew = np.zeros((0,3))
vnew = np.zeros((0,3))
for i in range(N):
    pos = np.loadtxt('pos/positions_a100_N10000_' + str(i))
    pnew = np.vstack((pnew, pos))

    bvel = np.loadtxt('vel/velocities_a100_N10000_' + str(i))
    vnew = np.vstack((vnew, bvel))

np.savetxt('pos/positions_a100_N' + str(nDM), pnew)
np.savetxt('vel/velocities_a100_N' + str(nDM), vnew)
