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
import time
from multiprocessing import Pool

lammpstrj_file_name = '300_dynamics.lammpstrj'
lammpstrj_file = open(lammpstrj_file_name,'r')
polymer_count = 11030
poly_indices = [int(i.split()[1])-1 for i in open('sys_evaporated.pdb','r').readlines() if 'ATOM' in i and int(i.split()[1])<=11030 and i.split()[-1] in ['O','C','N']]

for i in range(4):
    line = lammpstrj_file.readline()
atom_count = int(line)
water_count = int(atom_count-polymer_count)
water_molecule_count = int(water_count/3)

for i in range(2):
    line = lammpstrj_file.readline()
L = float(line.split()[-1])
lammpstrj_file.seek(0)

def go_to_line(file_name, line_index):
    ''' Bring the file to the indicated line. '''
    file_to_jump = open(file_name)
    if line_index != 0:
        next(itertools.islice(file_to_jump, line_index-1, None))
    return file_to_jump

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def calculate_director(molecule_coords):
    oxygen_coord = molecule_coords[0]
    vector = (molecule_coords[2]-oxygen_coord) + (molecule_coords[1]-oxygen_coord)
    return vector/np.linalg.norm(vector)

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

def dist_pbc(x0, x1, dimensions):
    '''Returns distance**2'''
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def minimum_distances_to_polymer(all_coords,water_oxygens,poly_indices,L): 
    distances = []
    poly_coords = all_coords[poly_indices]
    for oxygen_index in water_oxygens:
        coord = all_coords[oxygen_index]
        dists = dist_pbc(poly_coords, coord, L)**0.5
        distances.append(min(dists))
    return np.array(distances)

CORE_COUNT = 24
STARTS = [0]
WINDOW_SIZE = 20000

def get_mean_squared_displacement(water_group):

    all_msds = []
    all_distances = []
    for START_PT in STARTS:
        min_distance_array = np.zeros(len(water_group))
        line_index =  (atom_count+9) * (START_PT) + 9
        file_at_START_PT = go_to_line(lammpstrj_file_name,line_index)

        system_coords = np.zeros((atom_count,3))
        for atom_index in range(atom_count):
            line = file_at_START_PT.readline()[26:]
            system_coords[atom_index] = np.fromstring(line, sep=' ')
        system_coords = wrap_coords(system_coords, L)
        water_oxygens = list(range(atom_count))[polymer_count+min(water_group)*3:polymer_count+min(water_group)*3+3*len(water_group):3]

        min_distance_array = minimum_distances_to_polymer(system_coords,water_oxygens,poly_indices,L)


        all_distances.append(min_distance_array)

    for START_PT in STARTS:
        msd_array = np.zeros((len(water_group),WINDOW_SIZE))
        line_index =  (atom_count+9) * (START_PT) + (polymer_count + 9) + min(water_group) * 3

        file_at_START_PT = go_to_line(lammpstrj_file_name,line_index)

        for frame in range(WINDOW_SIZE):
            frame_coords = np.zeros((len(water_group)*3,3))
            for atom_index in range(len(water_group)*3):
                line = file_at_START_PT.readline()[26:]
                frame_coords[atom_index] = np.fromstring(line, sep=' ')
            frame_coords = wrap_coords(frame_coords,L)

            #only oxygen atoms
            frame_coords = frame_coords[::3]
            
            if frame != WINDOW_SIZE-1:
                for i in range(atom_count+9-len(water_group)*3):
                    next(file_at_START_PT)

            if frame == 0:
                initial_coords = copy.deepcopy(frame_coords)
                msd_array[:,0] = 0

            else:
                squared_displacement = dist_pbc(frame_coords,initial_coords,L)**0.5
                for iterindex,water_index in enumerate(water_group):
                
                    # water_oxygen_coord = frame_coords[iterindex]
                    # squared_displacement = dist_pbc(water_oxygen_coord,initial_coords[iterindex],L)
                    # if squared_displacement[iterindex] > 13:
                        # stack1 = np.vstack((frame_coords[iterindex]))
                    #     stack2 = initial_coords[iterindex] 
                        # print(frame_coords[iterindex],initial_coords[iterindex])
                        # exit()

                    msd_array[iterindex][frame] = squared_displacement[iterindex]
    

        all_msds.append(msd_array)

    stacked_msds = np.vstack(all_msds)
    stacked_distances = np.hstack(all_distances)

    return (stacked_msds,stacked_distances)

water_groups = split(list(range(water_molecule_count)),CORE_COUNT)

outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(get_mean_squared_displacement)(water_groups[i]) for i in range(CORE_COUNT))

all_msds = np.vstack([i[0] for i in outputs])
all_distances = np.hstack([i[1] for i in outputs])

f_out = open('msd.out','wb')
np.save(f_out,all_msds)
np.save(f_out,all_distances)
f_out.close()