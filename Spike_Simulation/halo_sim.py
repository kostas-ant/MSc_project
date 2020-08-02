#!/usr/bin/env python3

import numpy as np
from amuse.lab import Particles, units, nbody_system, ph4
import time
import argparse


root_directory = '/code/' # for loading and saving

parser = argparse.ArgumentParser(description='...')
parser.add_argument('-soft','--soft', help='softening length in pc', type=float, default=1)
parser.add_argument('-N_DM','--N_DM', help='Number of DM particles', type=int, default=10000)
parser.add_argument('-num_workers','--num_workers', help='Number of cores', type=int, default=2)
parser.add_argument('-t','--t', help='Simulation time in years', type=float, default=1e5)
args = parser.parse_args()

soft_length = args.soft

start = time.time()


def sim(nDM, t_end, softening, num_workers):
    pos = np.loadtxt(root_directory + 'pos/positions_a100_N' + str(nDM))
    vel = np.loadtxt(root_directory + 'vel/velocities_a100_N' + str(nDM))
    # m = np.loadtxt(root_directory + 'masses/mass_particle_a100_N' + str(nDM))
    M_halo = np.load(root_directory + 'M_halo_a100.npy')

    bodies = Particles(nDM)

    for i in range(nDM):
        p = bodies[i]
        p.position = pos[i,:] | units.pc
        p.velocity = vel[i,:] | (units.pc/units.s)
        p.mass = M_halo/nDM | units.MSun

    # Simulation
    convert_nbody = nbody_system.nbody_to_si(1|units.MSun, 1|units.pc)
    gravity = ph4(convert_nbody, number_of_workers=num_workers)
    gravity.particles.add_particles(bodies)
    # gravity.parameters.timestep_parameter = timestep
    gravity.parameters.epsilon_squared = softening*softening


    gravity.evolve_model(t_end)
    channel = gravity.particles.new_channel_to(bodies) # update bodies list
    channel.copy()

    gravity.stop()
    # header= 'soft_length=%.2f, no BH, t_end=%.1e' %(softening.value_in(units.pc), t_end.value_in(units.yr))
    # np.savetxt(root_directory + 'pos/positions_f_a100_N' + str(nDM), bodies.position[:,:].value_in(units.pc), header=header)
    # np.savetxt(root_directory + 'vel/velocities_f_a100_N' + str(nDM), bodies.velocity[:,:].value_in(units.pc/units.s), header=header)
    # np.savetxt(root_directory + 'masses/mass_p_f_a100_N' + str(nDM), bodies.mass[:].value_in(units.MSun),header=header)

    return bodies


nDM = args.N_DM # number of particles
num_work = args.num_workers
t_end = args.t | units.yr
# np.save(root_directory + 't_end', t_end.value_in(units.yr))

bodies_f = sim(nDM, t_end, soft_length | units.pc, num_work) # run Simulation

end = time.time()
print('---------------- elapsed time (min) = ', (end-start)/60)
