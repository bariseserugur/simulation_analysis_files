import numpy as np
from joblib import Parallel,delayed
import argparse
import sys
import time
import os
import pickle
import mdtraj as md

'''This code calculates either the number of changes in the nearest water molecule per grid or the number of unique nearest
water molecules to a grid center in 10 ps.'''

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def find_nearest_water(grid, water_coords, water_residues, L):
    nearest_waters = []
    for coord in grid: 
        dists = dist_pbc(water_coords, coord, L)**0.5
        if np.min(dists) < 2:
            nearest_waters.append(water_residues[np.argmin(dists)])
        else:
            nearest_waters.append(9999)
    return nearest_waters

#Obtain list of indices of water atoms
t = md.load('modified_300_dynamics.lammpstrj',top='modified_sys_evaporated.pdb')
waters = t.topology.select('resname BBB')
water_residues = [t.topology.atom(i).residue.index for i in waters]

#Box length
L = t[0].unitcell_lengths[0][0]

#load previously analyzed grid (occupied/unoccupied)
target_directory = os.getcwd()+'/'
read_directory = target_directory+'free_volume_outputs/'
read_name = 'coarse05_20.txt'
f_read = open(read_directory+read_name, "rb")
grid = pickle.load(f_read)

#filter grid only to display unoccupied grids
grid = grid[grid[:,3]==0][:,:3]
grids = split(grid,96)

st = time.time()

ncores = 96
#change_counts = np.zeros(len(grid))
#for frame,system_coords in enumerate(t.xyz[:500]):
#    water_coords = system_coords[waters] * 10 #in Angstroms
#    outputs = Parallel(n_jobs=ncores)(delayed(find_nearest_water)(grids[i],water_coords,water_residues,L) for i in range(ncores))
#    nearest_waters = np.hstack(outputs)
#    if frame > 0:
#        diff = nearest_waters-prev_nearest_waters
#        change_array = np.where(diff!=0,1,0)
#        change_counts += change_array
#    prev_nearest_waters = nearest_waters

uniques = [[] for i in grid]
for frame,system_coords in enumerate(t.xyz[:500]):
    water_coords = system_coords[waters] * 10 #in Angstroms
    outputs = Parallel(n_jobs=ncores)(delayed(find_nearest_water)(grids[i],water_coords,water_residues,L) for i in range(ncores))
    nearest_waters = np.hstack(outputs)
    for index,obj in enumerate(nearest_waters):
        if obj not in uniques[index]:
            uniques[index].append(obj)
change_counts = [len(i) for i in uniques]

mean_change_count = np.mean(change_counts)

f_out = open('water_flux.out','w')
f_out.write('water_flux = {}'.format(mean_change_count))
