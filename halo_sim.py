#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from amuse.lab import Particles, units, nbody_system, ph4
from mpl_toolkits.mplot3d import Axes3D
# import init_conditions
import time

start = time.time()

def sim(nDM, t_end, M_halo, softening):
    pos = np.loadtxt('/code/pos/positions_a1_N10000')
    vel = np.loadtxt('/code/vel/velocities_a1_N10000')
    m = np.loadtxt('/code/masses/mass_particle_a1_N10000')
    # add BH in the end
    # pos = np.vstack((posh, np.zeros(3)))
    # vel = np.vstack((velh, np.zeros(3)))
    # m = np.append(mh, 1e4)
    bodies = Particles(nDM)

    for i in range(nDM):
        p = bodies[i]
        p.position = pos[i,:] | units.pc
        p.velocity = vel[i,:] | (units.pc/units.s)
        p.mass = m[i] | units.MSun

    # Simulation
    convert_nbody = nbody_system.nbody_to_si(1|units.MSun, 1|units.pc)
    gravity = ph4(convert_nbody)
    gravity.particles.add_particles(bodies)
    # gravity.parameters.timestep_parameter = timestep
    gravity.parameters.epsilon_squared = softening*softening
    BH = gravity.particles[-1]

    # while gravity.model_time < t_end:
    #     gravity.evolve_model(gravity.model_time + (t_end/5))
    #     BH.mass += 1e4 | units.MSun

    gravity.evolve_model(t_end)
    channel = gravity.particles.new_channel_to(bodies) # update bodies list
    channel.copy()

    gravity.stop()
    header= 't_end=%.1e' %(t_end.value_in(units.yr))
    np.savetxt('/code/pos/positions_f_a1_N10000', bodies.position[:,:].value_in(units.pc), header=header)
    np.savetxt('/code/vel/velocities_f_a1_N10000', bodies.velocity[:,:].value_in(units.pc/units.s), header=header)
    np.savetxt('/code/masses/mass_p_f_a1_N10000', bodies.mass[:].value_in(units.MSun),header=header)
    return bodies


nDM = 10000 #np.load('/code/nDM.npy') #init_conditions.nDM
M_halo = np.load('/code/M_halo_a1_N10000.npy')
t_end = 1e6 | units.yr
np.save('/code/t_end', t_end.value_in(units.yr))
bodies = sim(nDM, t_end, M_halo, 1e-3 | units.pc)
end = time.time()
print('---------------- elapsed time (min) = ', (end-start)/60)
