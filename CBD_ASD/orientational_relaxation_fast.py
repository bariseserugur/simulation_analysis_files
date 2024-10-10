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
polymer_count = len([i for i in open('sys.pdb','r').readlines() if 'ATOM' in i])
poly_indices = [int(i.split()[1])-1 for i in open('sys.pdb','r').readlines() if 'ATOM' in i and i.split()[-1] in ['O','C']]

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

# with open('300_dynamics.lammpstrj') as f:
#     tinit = time.time()
#     for line in itertools.islice(f, 0, 10000000, atom_count+9):
#         print(line,'a')
#         print(f.readline(),'b')
#         # print(time.time()-tinit)
#     # break

# print(time.time()-tinit)


# tinit = time.time()
# for frame_index in range(500):
#     tinit = time.time()
#     frame_coords = np.zeros((atom_count,3))
#     for i in range(9):
#         lammpstrj_file.readline()
#     for i in range(atom_count):
#         line = lammpstrj_file.readline()[26:]
#         frame_coords[i] = np.fromstring(line, sep=' ')
#     # pickle.dump(frame_coords,pickle_file)
#     print(time.time()-tinit)
# exit()

# topology = md.load('polymer_water.pdb').topology
# atom_count = topology.n_atoms
# water_name = topology.atom(atom_count-1).residue.name
# water_count = int(len([i for i in topology.atoms if i.residue.name == water_name])/3)
# max_polymer_index = max([i.index for i in topology.atoms if i.residue.name != water_name])

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def calculate_director(molecule_coords):
    oxygen_coord = molecule_coords[0]
    vector = (molecule_coords[2]-oxygen_coord) + (molecule_coords[1]-oxygen_coord)
    return vector/np.linalg.norm(vector)

# def unwrap_coords(molecule_coords,L):
#     oxygen_coord = molecule_coords[0]
#     delta_coords = molecule_coords-oxygen_coord
#     return np.where(delta_coords > 0.5*L, molecule_coords-L, np.where(delta_coords < -0.5*L, molecule_coords+L, molecule_coords))

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

def dist_pbc(x0, x1, dimensions): 
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

CORE_COUNT = 96
STARTS = [0,5000]
WINDOW_SIZE = 5000
proximity_groups = [tuple([i,i+1]) for i in range(2,15)] + [(15,100)]

def get_autocorrelation_function(water_group):
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

    all_orientational_arrays = []
    for START_PT in STARTS:
        orientational_array = np.zeros((len(water_group),WINDOW_SIZE))
        line_index =  (atom_count+9) * (START_PT) + (polymer_count + 9) + min(water_group) * 3

        file_at_START_PT = go_to_line(lammpstrj_file_name,line_index)

        for frame in range(WINDOW_SIZE):
            frame_coords = np.zeros((len(water_group)*3,3))
            for atom_index in range(len(water_group)*3):
                line = file_at_START_PT.readline()[26:]
                frame_coords[atom_index] = np.fromstring(line, sep=' ')
            
            if frame != WINDOW_SIZE-1:
                for i in range(atom_count+9-len(water_group)*3):
                    next(file_at_START_PT)

            if frame == 0:
                init_orientations = np.zeros((len(water_group),3))
                for iterindex,water_index in enumerate(water_group):
                    water_coords = frame_coords[iterindex*3:(iterindex+1)*3]
                    init_orientations[iterindex] = calculate_director(water_coords)
                    orientational_array[:,0] = 1

            else:
                for iterindex,water_index in enumerate(water_group):
                    water_coords = frame_coords[iterindex*3:(iterindex+1)*3]
                    director = calculate_director(water_coords)
                    dot = np.dot(director,init_orientations[iterindex])
                    legendre = (3*(dot)**2 - 1)/2
                    orientational_array[iterindex][frame] = legendre
        all_orientational_arrays.append(orientational_array)

    stacked_orientations = np.vstack(all_orientational_arrays)
    stacked_distances = np.hstack(all_distances)

    autocorrelation_functions = {}
    proximity_populations = {}

    for proximity_group in proximity_groups:
        lower_bound = proximity_group[0]
        higher_bound = proximity_group[1]
        indices = np.where((lower_bound < stacked_distances) & (stacked_distances < higher_bound))
        autocorrelation_functions[lower_bound,higher_bound] = np.mean(stacked_orientations[indices],axis=0)
        proximity_populations[lower_bound,higher_bound] = len(indices[0])
    return (autocorrelation_functions, proximity_populations)

water_groups = split(list(range(water_molecule_count)),CORE_COUNT)

outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(get_autocorrelation_function)(water_groups[i]) for i in range(CORE_COUNT))

proximity_populations = {}
autocorrelation_functions = {}
for proximity_group in proximity_groups:
        lower_bound = proximity_group[0]
        higher_bound = proximity_group[1]
        proximity_populations[lower_bound,higher_bound] = 0
        autocorrelation_functions[lower_bound,higher_bound] = np.zeros(WINDOW_SIZE)

for proximity_group in proximity_groups:
    lower_bound = proximity_group[0]
    higher_bound = proximity_group[1]
    for core_number in range(CORE_COUNT):
        proximity_populations[lower_bound,higher_bound] += outputs[core_number][1][lower_bound,higher_bound]
        autocorrelation_functions[lower_bound,higher_bound] += outputs[core_number][0][lower_bound,higher_bound] * outputs[core_number][1][lower_bound,higher_bound]

for proximity_group in proximity_groups:
    lower_bound = proximity_group[0]
    higher_bound = proximity_group[1]
    autocorrelation_functions[lower_bound,higher_bound] /= proximity_populations[lower_bound,higher_bound]


# orientational_relaxation = np.vstack([i[0] for i in orientational_relaxation_outputs])
# distances = np.hstack([i[1] for i in orientational_relaxation_outputs])



f_out = open('orientational_relaxation.out','w')
# autocorrelation_functions = {}
# proximity_populations = {}
for proximity_group in proximity_groups:
    lower_bound = proximity_group[0]
    higher_bound = proximity_group[1]
    # indices = np.where((lower_bound < distances) & (distances < higher_bound))
    # autocorrelation_functions[lower_bound,higher_bound] = np.mean(orientational_relaxation[indices],axis=0)
    # proximity_populations[lower_bound,higher_bound] = len(indices[0])
    f_out.write('autocorrelation_functions[{},{}] = {}\n'.format(lower_bound,higher_bound,list(autocorrelation_functions[lower_bound,higher_bound])))
    f_out.write('proximity_populations[{},{}] = {}\n'.format(lower_bound,higher_bound,proximity_populations[lower_bound,higher_bound]))
f_out.close()