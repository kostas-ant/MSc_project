#!/usr/bin/env python3

import numpy as np
from amuse.lab import Particles, units, nbody_system, ph4
import time
from tqdm import tqdm
import argparse


root_directory = '/code/' # for loading and saving

parser = argparse.ArgumentParser(description='...')
parser.add_argument('-soft','--soft', help='softening length in pc', type=float, default=1)
parser.add_argument('-N_DM','--N_DM', help='Number of DM particles', type=int, default=100000)
parser.add_argument('-num_workers','--num_workers', help='Number of cores', type=int, default=16)
parser.add_argument('-t','--t', help='Simulation time in years', type=float, default=1e5)
parser.add_argument('-tgrow','--tgrow', help='Growth time of BH in years', type=float, default=1e4)
parser.add_argument('-step','--step', help='timestep to grow BH', type=int, default=100)
parser.add_argument('-a','--a', help='Scale radius', type=int, default=100)
parser.add_argument('-M_BH','--M_BH', help='BH mass', type=float, default=6e6)

args = parser.parse_args()

soft_length = args.soft
a = args.a

start = time.time()


def sim(nDM, t_end, t_grow, m_bh, timestep, softening, num_workers):
    posh = np.loadtxt(root_directory + 'pos/positions_a' +str(a)+ '_N' + str(nDM-1))
    velh = np.loadtxt(root_directory + 'vel/velocities_a' +str(a)+ '_N' + str(nDM-1))
    # mh = np.loadtxt(root_directory + 'masses/mass_particle_a100_N' + str(nDM-1))
    M_halo = np.load(root_directory + 'M_halo_a' +str(a)+ '.npy')

    # add BH in the end
    pos = np.vstack((posh, np.zeros(3)))
    vel = np.vstack((velh, np.zeros(3)))
    # m = np.append(mh, 1)
    bodies = Particles(nDM)

    for i in range(nDM):
        p = bodies[i]
        p.position = pos[i,:] | units.pc
        p.velocity = vel[i,:] | (units.pc/units.s)
        if i<(nDM-1) :
            p.mass = M_halo/(nDM-1) | units.MSun
        else:
            p.mass = 0 | units.MSun # initial BH mass

    # Simulation
    convert_nbody = nbody_system.nbody_to_si(1|units.MSun, 1|units.pc)
    gravity = ph4(convert_nbody, number_of_workers=num_workers)
    gravity.particles.add_particles(bodies)
    # gravity.parameters.timestep_parameter = timestep
    gravity.parameters.epsilon_squared = softening*softening
    BH = gravity.particles[-1]

    # save the positions in each timestep
    pos_stepx = [gravity.particles.x.value_in(units.pc)]
    pos_stepy = [gravity.particles.y.value_in(units.pc)]
    pos_stepz = [gravity.particles.z.value_in(units.pc)]

    pbar = tqdm(total=timestep)
    while gravity.model_time < t_grow:
        BH.mass += m_bh/timestep | units.MSun
        gravity.evolve_model(gravity.model_time + (t_grow/timestep))

        # pos_stepx.append(gravity.particles.x.value_in(units.pc))
        # pos_stepy.append(gravity.particles.y.value_in(units.pc))
        # pos_stepz.append(gravity.particles.z.value_in(units.pc))

        pbar.update(1)
    pbar.close()

    gravity.evolve_model(t_end)
    channel = gravity.particles.new_channel_to(bodies) # update bodies list
    channel.copy()

    gravity.stop()
    header= 'soft_length=%.2f, t_grow=%.2e, t_end=%.1e, M_bh=%.2e' %(softening.value_in(units.pc),t_grow.value_in(units.yr), t_end.value_in(units.yr), m_bh)
    np.savetxt(root_directory + 'pos/positions_f_a' +str(a)+ '_N' + str(nDM-1) + '_bh' + str(int(m_bh/1e6)) +'_'+ str(timestep), bodies.position[:,:].value_in(units.pc), header=header)
    np.savetxt(root_directory + 'vel/velocities_f_a' +str(a)+ '_N' + str(nDM-1) + '_bh' + str(int(m_bh/1e6)) +'_'+ str(timestep), bodies.velocity[:,:].value_in(units.pc/units.s), header=header)
    # np.savetxt(root_directory + 'masses/mass_p_f_a' +str(a)+ '_N' + str(nDM-1) + '_bh' + str(int(m_bh/1e6)), bodies.mass[:].value_in(units.MSun),header=header)

    # np.savetxt(root_directory + 'pos/step/positions_f_a100_N' + str(nDM-1) + 'stepx', pos_stepz)
    # np.savetxt(root_directory + 'pos/step/positions_f_a100_N' + str(nDM-1) + 'stepy', pos_stepy)
    # np.savetxt(root_directory + 'pos/step/positions_f_a100_N' + str(nDM-1) + 'stepz', pos_stepz)

    return bodies


nDM = args.N_DM # number of particles
num_work = args.num_workers
t_end = args.t | units.yr
t_growth = args.tgrow | units.yr
t_step = args.step
M_bh = args.M_BH
# np.save(root_directory + 't_end', t_end.value_in(units.yr))

bodies_f = sim(nDM+1, t_end, t_growth, M_bh, t_step, soft_length | units.pc, num_work) # run Simulation, nDM+1 to add the BH

end = time.time()
print('---------------- elapsed time (min) = ', (end-start)/60)
