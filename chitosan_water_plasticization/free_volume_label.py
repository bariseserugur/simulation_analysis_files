import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import argparse
import sys
import time
import os
import pickle
import shutil
from label_monomers import label_monomers
from numba import jit
import numexpr as ne
import numba as nb

'''Calculate the free volume at different temperatures using the saved configurations.'''

temperatures = []
for i in os.listdir():
    if 'equil.save' in i:
        temperatures.append(int(i.split('_')[0]))
temperatures = sorted(temperatures)

# temperatures = [200]

#Read sys.pdb file
atom_names = []
fid = open('sys_evaporated.pdb','r').readlines()
for line in fid:
    if 'ATOM' in line:
        atom_names.append(line.split()[2])
    if 'CONECT' in line:
        break

#Read ff_sys.xml file
name_to_type = {}
type_to_sigma = {}
fid = open('ff_sys.xml','r').readlines()
for line in fid:
    if '<Atom name=' in line:
        name_to_type[line.split('"')[1]] = line.split('"')[3]
    if '<Atom type=' in line:
        type_to_sigma[line.split('"')[1]] = 10*float(line.split('"')[5]) #in Angstroms
atom_sigmas = np.array([type_to_sigma[name_to_type[i]] for i in atom_names])
atom_sigmas = np.array([i/2 if i!=934189 else 1.5 for i in atom_sigmas]) ######################################################################CHANGE
atom_count = len(atom_sigmas)

monomer_water_groups = [[] for i in range(500)] #500 monomers
monomers_dict = label_monomers()
for i in np.arange(11030):
    monomer_number = monomers_dict[i]- 1
    monomer_water_groups[monomer_number].append(i)

#Only include bound water molecules
# exec(open('bound_water_indices.out','r').readlines()[0])

# for index,i in enumerate(np.arange(11030,atom_count,3)):
#     if index in bound_water_indices:
#         water_indices = [i,i+1,i+2]
#         monomer_water_groups.append(water_indices)
# monomer_water_groups = [np.array(i) for i in monomer_water_groups]

for i in np.arange(11030,atom_count,3):
    water_indices = [i,i+1,i+2]
    monomer_water_groups.append(water_indices)
monomer_water_groups = [np.array(i) for i in monomer_water_groups]

def read_positions(frame_range):
    file_name = '300_dynamics.lammpstrj'
    lammpstrj_file = open(file_name,'r')
    for i in range(4):
        line = lammpstrj_file.readline()
    atom_count = int(line)
    lammpstrj_file.seek(0)

    xyz_list = []
    L_list = []
    frame_range = sorted(frame_range)
    current_frame = 0
    for index,frame in enumerate(frame_range):
        print(frame)
        frame_coords = np.zeros((atom_count,3))
        frames_to_jump = int(frame-current_frame)
        #jump frames
        for i in range(frames_to_jump * (atom_count+9)):
            next(lammpstrj_file)
        for i in range(5):
            next(lammpstrj_file)
        L = float(lammpstrj_file.readline().split()[1])
        for i in range(3):
            next(lammpstrj_file)
        for i in range(atom_count):
            line = lammpstrj_file.readline()[26:]
            frame_coords[i] = np.fromstring(line, sep=' ')
        current_frame = frame+1
        xyz_list.append(frame_coords)
        L_list.append(L)
    return xyz_list,L_list

@jit(nopython=True)
def find_overlaps(R_PROBE, grid, system_coords, atom_sigmas, L, system_indices): 
    # calculate distance from polymer molecules
    # if distance is < r_probe + r_polymer: overlaps
    overlaps = np.zeros(len(grid))
    overlap_indices = np.zeros(len(grid)) - 1
    intersection_radius = R_PROBE + atom_sigmas

    for grid_index,grid_coord in enumerate(grid):
        # calculates dist_pbc between probe coord and all atom positions
        # then checks if any of these distances is less than r_probe+r_atom
        distances_with_sys = dist_pbc(system_coords, grid_coord, L)-intersection_radius
        free_volume_check = distances_with_sys < 0
        if np.any(free_volume_check):
            overlaps[grid_index] = 1
            first_fill_index = system_indices[np.where(free_volume_check)[0][0]]
            if first_fill_index < 11030:
                overlap_indices[grid_index] = 1
            else:
                overlap_indices[grid_index] = first_fill_index
    return np.hstack((grid[overlaps==1],overlap_indices[overlaps==1].reshape(-1,1)))

@jit(nopython=True)
def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.sqrt(np.sum(delta**2, axis=1))

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def group_overlap_check(xyz,L,R_PROBE,list_of_groups):
    all_overlaps = np.array([99999,99999,99999,99999])
    for atom_index_group in list_of_groups:
        group_positions = xyz[atom_index_group]
        group_sigmas = atom_sigmas[atom_index_group]
        
        xrange = gridx[(gridx<(np.max(group_positions[:,0]+group_sigmas)+R_PROBE)) &  (gridx>(np.min(group_positions[:,0]-group_sigmas)-R_PROBE))]
        xrange = np.where(xrange > grid_L, xrange-grid_L, np.where(xrange < 0, xrange+grid_L, xrange))
        yrange = gridy[(gridy<(np.max(group_positions[:,1]+group_sigmas)+R_PROBE)) &  (gridy>(np.min(group_positions[:,1]-group_sigmas)-R_PROBE))]
        yrange = np.where(yrange > grid_L, yrange-grid_L, np.where(yrange < 0, yrange+grid_L, yrange))
        zrange = gridz[(gridz<(np.max(group_positions[:,2]+group_sigmas)+R_PROBE)) &  (gridz>(np.min(group_positions[:,2]-group_sigmas)-R_PROBE))]
        zrange = np.where(zrange > grid_L, zrange-grid_L, np.where(zrange < 0, zrange+grid_L, zrange))

        grid = np.array(np.meshgrid(xrange,yrange,zrange)).T.reshape(-1, 3)
        overlaps = find_overlaps(R_PROBE,grid,xyz[atom_index_group],atom_sigmas[atom_index_group],L,atom_index_group)

        all_overlaps = np.vstack((all_overlaps,overlaps))
    all_overlaps = np.unique(all_overlaps.round(decimals=1)[1:],axis=0)
    return all_overlaps

R_PROBE = 0.1
CORE_COUNT = 24

monomers_split = split(monomer_water_groups[:500],CORE_COUNT)
waters_split = split(monomer_water_groups[500:],CORE_COUNT)
final_groups = [(obj+waters_split[index]) for index,obj in enumerate(monomers_split)]

xyz_list = []
L_list = []
for temperature in temperatures:
    save_file = open('{}_equil.save'.format(temperature))
    save_file.readline()
    L_list.append(float(save_file.readline().split()[0])*10)
    for i in range(3):
        save_file.readline()
    temperature_coords = np.zeros((atom_count,3))
    for i in range(atom_count):
        line = save_file.readline()
        temperature_coords[i] = np.fromstring(line,sep=' ') * 10
    xyz_list.append(temperature_coords)

free_volume_fractions = []
for temperature_index,temperature in enumerate(temperatures):
    print(temperature)
    init = time.time()

    L = L_list[temperature_index]
    xyz = xyz_list[temperature_index]
    wrapped_xyz = wrap_coords(xyz,L)

    #the grids are extra long
    gridx = np.array(sorted(np.hstack([np.arange(0, L+100, 2*R_PROBE),-np.arange(0, L+100, 2*R_PROBE)[1:]])))
    gridy = np.array(sorted(np.hstack([np.arange(0, L+100, 2*R_PROBE),-np.arange(0, L+100, 2*R_PROBE)[1:]])))
    gridz = np.array(sorted(np.hstack([np.arange(0, L+100, 2*R_PROBE),-np.arange(0, L+100, 2*R_PROBE)[1:]])))
    grid_L = max(np.arange(0, L, 2*R_PROBE))   

    parallel_outputs = Parallel(n_jobs=CORE_COUNT)(delayed(group_overlap_check)(wrapped_xyz,L,R_PROBE,final_groups[i]) for i in range(CORE_COUNT))

    actual_grid = (np.array(np.meshgrid(np.arange(0, L, 2*R_PROBE).round(decimals=1),np.arange(0, L, 2*R_PROBE).round(decimals=1),np.arange(0, L, 2*R_PROBE).round(decimals=1))).T.reshape(-1, 3))

    print(len(actual_grid))

    filled_grids_4d = np.vstack(parallel_outputs)

    t = time.time()
    filled_grids_4d = filled_grids_4d[filled_grids_4d[:, 3].argsort()]
    filled_grids_4d = filled_grids_4d[filled_grids_4d[:, 2].argsort()]
    filled_grids_4d = filled_grids_4d[filled_grids_4d[:, 1].argsort()]
    filled_grids_4d = filled_grids_4d[filled_grids_4d[:, 0].argsort()]

    filled_grids_4d = np.unique(filled_grids_4d,axis=0)

    filled_grids, duplicate_indices = np.unique(filled_grids_4d[:,[0,1,2]],axis=0,return_inverse=True)
    unq, count = np.unique(duplicate_indices, return_counts=True)

    indices_to_pick = []
    idx_counter = 0
    for index,obj in enumerate(count):
        indices_to_pick.append(idx_counter)
        idx_counter+=obj

    filled_grids_4d = filled_grids_4d[indices_to_pick]
    
    # filled_grids_final = []
    # for i in range(len(duplicate_indices)):
    #     lowest_index_grid = filled_grids_4d[duplicate_indices==i][0]
    #     filled_grids_final.append(lowest_index_grid)
    # filled_grids = np.array(filled_grids_final)
    # print(time.time()-t)

    # filled_grids_unique = np.unique(filled_grids.view(np.dtype((np.void, filled_grids.dtype.itemsize*filled_grids.shape[1])))).view(filled_grids.dtype).reshape(-1, filled_grids.shape[1])

    # filled_grids_unique = np.unique(filled_grids.view(np.dtype((np.void, filled_grids.dtype.itemsize*filled_grids.shape[1])))).view(filled_grids.dtype).reshape(-1, filled_grids.shape[1])
    # filled_grids_unique = set([tuple(i) for i in filled_grids_unique])
    # actual_grid_tuple = [tuple(i) for i in actual_grid]
    # print(time.time()-init)

    # @jit(nopython=True)
    # def setdiff2d_idx(arr1, arr2):
    #     idx = np.array([i not in arr2 for i in arr1])
    #     arr1 = np.array(arr1)
    #     return actual_grid[idx]

    def setdiff2d_idx(arr1, arr2):
        delta = set(map(tuple, arr2))
        idx = [tuple(x) not in delta for x in arr1]
        return arr1[idx]

    # empty_grids = setdiff2d_idx(actual_grid,filled_grids)

    free_volume_fraction = 1 - len(filled_grids_4d)/len(actual_grid)

    print(free_volume_fraction)
    
    print(len(actual_grid))

    f_out = open('free_volume_label_{}.out'.format(temperature),'wb')
    np.save(f_out,free_volume_fraction)
    np.save(f_out,filled_grids_4d[:,3])
    f_out.close()

    print(time.time()-init)
