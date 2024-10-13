import numpy as np
import sys
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.waterdynamics import WaterOrientationalRelaxation as WOR
import pickle
import joblib
from joblib import Parallel,delayed
import mdtraj as md
import copy
import itertools

def read_positions(frame_range):
    u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs
    L = u.trajectory[0].triclinic_dimensions[0][0] #constant L since NVE ensemble
    xyz_list = []
    for frame in frame_range:
        frame_xyz = u.trajectory[frame].positions
        xyz_list.append(copy.deepcopy(frame_xyz))
    return xyz_list,L

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

STARTS = [0]
WINDOW_SIZE = 20000
CORE_COUNT = 24
frames = np.arange(0,20000,1) ###CHANGE
split_frames = split(frames,CORE_COUNT)
xyz_outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(read_positions)(split_frames[i]) for i in range(CORE_COUNT))
L = xyz_outputs[0][1]
xyz_outputs = [i[0] for i in xyz_outputs]
xyz_outputs = list(itertools.chain(*xyz_outputs)) #merge list of lists
xyz = np.stack(xyz_outputs)
water_count = int((len(xyz[0]) - 11030)/3)

def get_mean_squared_displacement():
    msd_list = [[] for i in range(water_count)]
    for START_PT in STARTS:
        init_frame = copy.deepcopy(xyz[START_PT][11030::3]) #only oxygens
        for frame in range(WINDOW_SIZE):
            frame_coords = xyz[START_PT+frame][11030::3]
            delta = dist_pbc(frame_coords,init_frame,L)
            delta_squared = delta**2
            for index,obj in enumerate(delta_squared):
                msd_list[index].append(obj)
    return msd_list

print('started')
msd_list = get_mean_squared_displacement()

f_out = open('msd.out','w')
f_out.write('msd_list = {}\n'.format(msd_list))
f_out.close()