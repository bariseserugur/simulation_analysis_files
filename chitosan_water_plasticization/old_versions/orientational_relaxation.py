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

def calculate_director(molecule_coords):
    oxygen_coord = molecule_coords[0]
    vector = (molecule_coords[2]-oxygen_coord) + ((molecule_coords[1]-oxygen_coord))/2
    return vector/np.linalg.norm(vector)

def unwrap_coords(molecule_coords,L):
    oxygen_coord = molecule_coords[0]
    delta_coords = molecule_coords-oxygen_coord
    return np.where(delta_coords > 0.5*L, molecule_coords-L, np.where(delta_coords < -0.5*L, molecule_coords+L, molecule_coords))

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


STARTS = [0,5000,10000,15000]
WINDOW_SIZE = 5000
CORE_COUNT = 24
frames = np.arange(0,20000,1) ###CHANGE
split_frames = split(frames,CORE_COUNT)
xyz_outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(read_positions)(split_frames[i]) for i in range(CORE_COUNT))
L = xyz_outputs[0][1]
xyz_outputs = [i[0] for i in xyz_outputs]
xyz_outputs = list(itertools.chain(*xyz_outputs)) #merge list of lists
xyz = np.stack(xyz_outputs)
water_count = int((len(xyz[0]) - 11030)/3)

def get_autocorrelation_function(water_group_dict):
    orientational_list = []
    for START_PT in STARTS:
        water_group = water_group_dict[START_PT]
        orientations = np.zeros((WINDOW_SIZE,len(water_group),3))
        for frame in range(WINDOW_SIZE):
            frame_coords = xyz[START_PT+frame][11030::]
            for iterindex,water_index in enumerate(water_group):
                water_coords = frame_coords[water_index*3:(water_index+1)*3]
                unwrapped_water = unwrap_coords(water_coords,L)
                orientations[frame][iterindex] = calculate_director(unwrapped_water)

        orientational_relaxation = np.zeros((len(water_group),WINDOW_SIZE))
        init_frame = orientations[0]
        for iterindex,water_index in enumerate(water_group):
            for frame in range(WINDOW_SIZE):
                water_orientation = orientations[frame][iterindex]
                init_orientation = init_frame[iterindex]
                dot = np.dot(water_orientation,init_orientation)
                legendre = (3*(dot)**2 - 1)/2
                orientational_relaxation[iterindex][frame] = legendre
        orientational_list.append(orientational_relaxation)
    return np.vstack(orientational_list)


water_groups_dict = {}
for i in range(CORE_COUNT):
    water_groups_dict[i] = {}
for START_PT in STARTS:
    for i in range(CORE_COUNT):
        water_groups_dict[i][START_PT] = split(list(range(water_count)),CORE_COUNT)[i]

orientational_relaxation_outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(get_autocorrelation_function)(water_groups_dict[i]) for i in range(CORE_COUNT))
orientational_relaxation = np.vstack(orientational_relaxation_outputs)
autocorrelation_function = list(np.mean(orientational_relaxation,axis=0))

f_out = open('orientational_relaxation.out','w')
f_out.write('all_orientational_relaxation = {}\n'.format(autocorrelation_function))

topology = md.load('sys_evaporated.pdb').topology
water_oxygens = [i.index for i in topology.atoms if i.index > 11029 and i.element.symbol == 'O']
poly_indices = [i.index for i in topology.atoms if i.index < 11030 and i.element.symbol in ['O','C']]

polymer_waters = {}
non_polymer_waters = {}
for START_PT in STARTS:
    all_coords = xyz[START_PT]
    minimum_distances = minimum_distances_to_polymer(all_coords,water_oxygens,poly_indices,L)
    polymer_waters[START_PT] = np.where(minimum_distances<3.5)[0]
    non_polymer_waters[START_PT] = np.where(minimum_distances>3.5)[0]


polymer_waters_dict = {}
for i in range(CORE_COUNT):
    polymer_waters_dict[i] = {}
for START_PT in STARTS:
    for i in range(CORE_COUNT):
        polymer_waters_dict[i][START_PT] = split(polymer_waters[START_PT],CORE_COUNT)[i]

orientational_relaxation_outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(get_autocorrelation_function)(polymer_waters_dict[i]) for i in range(CORE_COUNT))
orientational_relaxation = np.vstack(orientational_relaxation_outputs)
autocorrelation_function = list(np.mean(orientational_relaxation,axis=0))
f_out.write('polymer_water_orientational_relaxation = {}\n'.format(autocorrelation_function))


non_polymer_waters_dict = {}
for i in range(CORE_COUNT):
    non_polymer_waters_dict[i] = {}
for START_PT in STARTS:
    for i in range(CORE_COUNT):
        non_polymer_waters_dict[i][START_PT] = split(non_polymer_waters[START_PT],CORE_COUNT)[i]

orientational_relaxation_outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(get_autocorrelation_function)(non_polymer_waters_dict[i]) for i in range(CORE_COUNT))
orientational_relaxation = np.vstack(orientational_relaxation_outputs)
autocorrelation_function = list(np.mean(orientational_relaxation,axis=0))
f_out.write('non_polymer_water_orientational_relaxation = {}\n'.format(autocorrelation_function))
f_out.close()