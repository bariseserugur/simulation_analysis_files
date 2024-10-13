import os
import numpy as np
import time
from joblib import Parallel,delayed
import itertools
import sys
from itertools import combinations
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.distances import dist
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import MDAnalysis as mda
import pickle
from numpy.linalg import norm
import copy
import joblib
from joblib import Parallel,delayed

'''Calculate Debye-Waller factor for water molecules.'''

sttime = time.time()

solvent_HB = {'water':['62','63'],'glycerol':['62','63','64','65','66','67']}

SOLVENT = str(sys.argv[1])
solvent_HB_types = solvent_HB[SOLVENT]
lammpstrj_file_name = '300_dynamics.lammpstrj'

lammpstrj_file = open(lammpstrj_file_name,'r')
for i in range(4):
    line = lammpstrj_file.readline()
atom_count = int(line)

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def go_to_line(file_name, line_index):
    ''' Bring the file to the indicated line. '''
    file_to_jump = open(file_name)
    if line_index != 0:
        next(itertools.islice(file_to_jump, line_index-1, None))
    return file_to_jump

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def minimum_distances_to_polymer(all_coords,water_oxygens,poly_indices,L): 
    distances = []
    poly_coords = all_coords[poly_indices]
    for oxygen_index in water_oxygens:
        coord = all_coords[oxygen_index]
        dists = dist_pbc(poly_coords, coord, L)**0.5
        distances.append(min(dists))
    return np.array(distances)

# u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs
# L = copy.deepcopy(u.trajectory[0].triclinic_dimensions[0][0]) #constant L since NVE ensemble

# def read_positions(frame_range):
#     u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs
#     xyz_list = []
#     for frame in frame_range:
#         frame_xyz = u.trajectory[frame].positions
#         xyz_list.append(copy.deepcopy(frame_xyz))
#     return xyz_list

def read_positions(FRAMES):
    '''Loads up the trajectory within frames onto memory'''

    system_coords = np.zeros((len(FRAMES),atom_count,3))

    line_index =  (atom_count+9) * (FRAMES[0]) + 9

    file_at_START_PT = go_to_line(lammpstrj_file_name,line_index)

    for frame_index,frame in enumerate(FRAMES):
        frame_coords = np.zeros((atom_count,3))

        for atom_index in range(atom_count):
            line = file_at_START_PT.readline()[26:]
            frame_coords[atom_index] = np.fromstring(line, sep=' ')
        
        system_coords[frame_index] = frame_coords

        if frame != max(FRAMES):
            for i in range(9):
                next(file_at_START_PT)
        
    return system_coords

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

topology = md.load('sys_evaporated.pdb').topology
water_oxygens = [i.index for i in topology.atoms if i.index > 11029 and i.element.symbol == 'O']

poly_indices = [int(i.split()[1])-1 for i in open('sys_evaporated.pdb','r').readlines() if 'ATOM' in i and int(i.split()[1])<=11030 and i.split()[-1] in ['O','C','N']]

lammpstrj_file = open(lammpstrj_file_name,'r')
for i in range(6):
    line = lammpstrj_file.readline()
L = float(line.split()[1])

CORE_COUNT = 24
STARTS_PER_CORE = 4

frames = list(np.arange(0,20000,1)) ###CHANGE
split_frames = split(frames,CORE_COUNT)

xyz_outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(read_positions)(split_frames[i]) for i in range(CORE_COUNT))

xyz_list_og = np.vstack(xyz_outputs)

# xyz_list = [item for sublist in xyz_outputs for item in sublist]
# xyz_list_og = np.stack(xyz_list)

WINDOW_SIZE = 5000
def run_DWF(start_frames):
    all_DWFs = []
    all_distances = []

    for start_frame in start_frames:
        start_xyz = xyz_list_og[start_frame]

        distances = minimum_distances_to_polymer(wrap_coords(start_xyz,L),water_oxygens,poly_indices,L)

        xyz = xyz_list_og[start_frame:start_frame+WINDOW_SIZE]
        xyzdelta = xyz-xyz[0]
        xyz = np.where(xyzdelta > 0.5*L, xyz-L, np.where(xyzdelta < -0.5*L, xyz+L, xyz))

        mean_pos = np.sum(xyz,axis=0)/WINDOW_SIZE
        DWFs = [[] for i in water_oxygens]
        for frame in range(len(xyz)):
            delta = xyz[frame]-mean_pos
            delta = np.where(delta > 0.5*L, delta-L, np.where(delta < -0.5*L, delta+L, delta))
            delta = np.sum((delta)**2,axis=1)
            for index,residueindex in enumerate(water_oxygens):
                DWFs[index].append(np.mean(delta[residueindex:residueindex+3],axis=0))
        all_DWFs.extend([np.mean(i) for i in DWFs])
        all_distances.extend(distances)
    return (all_DWFs,all_distances)

start_frames = [int(i) for i in np.linspace(0,len(frames)-WINDOW_SIZE,STARTS_PER_CORE*CORE_COUNT)]
split_frames = split(start_frames,CORE_COUNT)

# outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(run_DWF)(split_frames[i]) for i in range(CORE_COUNT))
# out_distances = [i[1] for i in outputs]
# out_DWFs = [i[0] for i in outputs]
# out_distances = [item for sublist in out_distances for item in sublist]
# out_DWFs = [item for sublist in out_DWFs for item in sublist]

out_DWFs, out_distances = run_DWF([0])
out_DWFs = np.array(out_DWFs)
out_distances = np.array(out_distances)

f_out = open('water_DWF.out','wb')
np.save(f_out,out_DWFs)
np.save(f_out,out_distances)
f_out.close()

print(time.time()-sttime)