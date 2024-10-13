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

'''This code yields a grid array (N,4) that indicates the location of the grid, and whether it is occupied or not.
It runs in parallel using 96 cores via joblib.'''

R_PROBE = float(sys.argv[1])
CORE_COUNT = 24

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
atom_sigmas = np.array([i/2 if index<11030 else 0 for index,i in enumerate(atom_sigmas)]) ######################################################################CHANGE

atom_count = len(atom_sigmas)

monomer_water_groups = [[] for i in range(500)] #500 monomers
monomers_dict = label_monomers()
for i in np.arange(11030):
    monomer_number = monomers_dict[i]- 1
    monomer_water_groups[monomer_number].append(i)

for i in np.arange(11030,atom_count,3):
    water_indices = [i,i+1,i+2]
    monomer_water_groups.append(water_indices)
monomer_water_groups = [np.array(i) for i in monomer_water_groups]

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

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

# def find_overlaps(R_PROBE, grid, system_coords, atom_sigmas, L): 
#     # calculate distance from polymer molecules
#     # if distance is < r_probe + r_polymer: overlaps
#     overlaps = np.zeros(len(grid))
#     l = 0
#     for coord in grid: 
#         # calculates dist_pbc between probe coord and all atom positions
#         # then checks if any of these distances is less than r_probe+r_atom
#         # flips overlaps[probe idx] if any of these distances is.
#         dists = dist_pbc(system_coords, coord, L)**0.5 < (R_PROBE + atom_sigmas)
#         if np.any(dists): 
#             overlaps[l] = 1
#         l += 1
#     return overlaps

def find_overlaps(R_PROBE, grid, system_coords, atom_sigmas, L): 
    # calculate distance from polymer molecules
    # if distance is < r_probe + r_polymer: overlaps
    overlaps = np.zeros(len(grid))
    l = 0
    for coord in grid: 
        # calculates dist_pbc between probe coord and all atom positions
        # then checks if any of these distances is less than r_probe+r_atom
        # flips overlaps[probe idx] if any of these distances is.
        dists = dist_pbc(system_coords, coord, L)**0.5 < (R_PROBE + atom_sigmas) 
        if np.any(dists): 
            overlaps[l] = 1
        l += 1
    return overlaps

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def group_overlap_check(xyz,L,R_PROBE,list_of_groups):
    all_overlaps = np.array([999,999,999])
    ctr = 0
    for atom_index_group in list_of_groups:
        # print(ctr)
        ctr+=1
        group_sigmas = atom_sigmas[atom_index_group]
        group_positions = xyz[atom_index_group][group_sigmas!=0]
        group_sigmas = group_sigmas[group_sigmas!=0]
        if len(group_sigmas) == 0:
            continue

        xrange = gridx[(gridx<(np.max(group_positions[:,0]+group_sigmas)+R_PROBE)) &  (gridx>(np.min(group_positions[:,0]-group_sigmas)-R_PROBE))]
        xrange = np.where(xrange > grid_L, xrange-grid_L, np.where(xrange < 0, xrange+grid_L, xrange))
        yrange = gridy[(gridy<(np.max(group_positions[:,1]+group_sigmas)+R_PROBE)) &  (gridy>(np.min(group_positions[:,1]-group_sigmas)-R_PROBE))]
        yrange = np.where(yrange > grid_L, yrange-grid_L, np.where(yrange < 0, yrange+grid_L, yrange))
        zrange = gridz[(gridz<(np.max(group_positions[:,2]+group_sigmas)+R_PROBE)) &  (gridz>(np.min(group_positions[:,2]-group_sigmas)-R_PROBE))]
        zrange = np.where(zrange > grid_L, zrange-grid_L, np.where(zrange < 0, zrange+grid_L, zrange))

        grid = np.array(np.meshgrid(xrange,yrange,zrange)).T.reshape(-1, 3)
        
        overlaps = grid[find_overlaps(R_PROBE,grid,wrapped_xyz[atom_index_group],atom_sigmas[atom_index_group],L) == 1]
        all_overlaps = np.vstack((all_overlaps,overlaps))
    all_overlaps = np.unique(all_overlaps.round(decimals=1)[1:],axis=0)
    return all_overlaps

frame_range = [0] #np.arange(0,10,1)
xyz_list,L_list = read_positions(frame_range)

monomers_split = split(monomer_water_groups[:500],CORE_COUNT)
waters_split = split(monomer_water_groups[500:],CORE_COUNT)
final_groups = [(obj+waters_split[index]) for index,obj in enumerate(monomers_split)]

free_volume_fractions = []
for frame_index,frame in enumerate(frame_range):
    init = time.time()

    L = L_list[frame_index]
    xyz = xyz_list[frame_index]
    wrapped_xyz = wrap_coords(xyz,L)

    #the grids are extra long
    gridx = np.array(sorted(np.hstack([np.arange(0, L+100, 2*R_PROBE),-np.arange(0, L+100, 2*R_PROBE)[1:]])))
    gridy = np.array(sorted(np.hstack([np.arange(0, L+100, 2*R_PROBE),-np.arange(0, L+100, 2*R_PROBE)[1:]])))
    gridz = np.array(sorted(np.hstack([np.arange(0, L+100, 2*R_PROBE),-np.arange(0, L+100, 2*R_PROBE)[1:]])))
    grid_L = max(np.arange(0, L, 2*R_PROBE))   

    parallel_outputs = Parallel(n_jobs=CORE_COUNT)(delayed(group_overlap_check)(wrapped_xyz,L,R_PROBE,final_groups[i]) for i in range(CORE_COUNT))

    actual_grid = (np.array(np.meshgrid(np.arange(0, L, 2*R_PROBE).round(decimals=1),np.arange(0, L, 2*R_PROBE).round(decimals=1),np.arange(0, L, 2*R_PROBE).round(decimals=1))).T.reshape(-1, 3))

    #since multiple groups can occupy the same volume, filled_grids has many repeating elements
    filled_grids = np.vstack(parallel_outputs)
    # filled_grids = set(map(tuple, np.vstack(parallel_outputs)))

    def setdiff2d_idx(arr1, arr2):
        delta = set(map(tuple, arr2))
        idx = [tuple(x) not in delta for x in arr1]
        return arr1[idx]
    init = time.time()
    empty_grids = setdiff2d_idx(actual_grid,filled_grids)
    free_volume_fraction = len(empty_grids)/len(actual_grid)

    f_out = open('fvf{}.out'.format(R_PROBE),'wb')
    np.save(f_out,free_volume_fraction)
    np.save(f_out,empty_grids)
    np.save(f_out,filled_grids) #with repeating elements
    f_out.close()
    print(time.time()-init)