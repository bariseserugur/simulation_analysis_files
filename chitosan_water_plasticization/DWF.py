import os
import numpy as np
import time
import itertools
import sys
import mdtraj as md

t = md.load('modified_300_dynamics.lammpstrj',top='modified_sys_evaporated.pdb')
xyz = t.xyz
L = t[0].unitcell_lengths[0][0]

'''if a particle crosses the simulation box, it moves from ~0 to 40 Angstrom, and the mean position will be ~20 Angstrom.
To fix this, adjust all xyz values so that they are on the same side of the box with the first coordinate in the trajectory.'''
xyzdelta = xyz-xyz[0]
xyz = np.where(xyzdelta > 0.5*L, xyz-L, np.where(xyzdelta < -0.5*L, xyz+L, xyz))
mean_pos = np.sum(xyz,axis=0)/len(t)

one_polymer_segments = [np.array(i) for i in [range(222),range(222,442),range(442,662),range(662,882),range(882,1103)]]
all_polymer_segments = []
for poly_no in range(10):
    all_polymer_segments.extend([i+poly_no*1103 for i in one_polymer_segments])

segment_DWF = [[] for i in all_polymer_segments]
for frame in range(len(xyz)):
    # print(frame)
    delta = xyz[frame]-mean_pos
    delta = np.where(delta > 0.5*L, delta-L, np.where(delta < -0.5*L, delta+L, delta))
    delta = np.sum((delta)**2,axis=1)
    for segment_no,segment_indices in enumerate(all_polymer_segments):
        segment_DWF[segment_no].append(np.mean(delta[segment_indices],axis=0))

f_out = open('DWF.out','w')
f_out.write('{}'.format([np.mean(i) for i in segment_DWF]))
f_out.close()