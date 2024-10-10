import os
import numpy as np
import time
from joblib import Parallel,delayed
import itertools
import sys
from itertools import combinations
import mdtraj as md
import MDAnalysis as mda
import copy
from return_monomer import return_monomer
from scipy.stats import linregress

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.sqrt(np.sum(delta**2))

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

topology = md.load('sys.pdb').topology

POLY = str(sys.argv[1]).upper()

atom_index_to_monomer_index = return_monomer(POLY)
monomer_index_to_chain_index = {}

for atom_index in atom_index_to_monomer_index.keys():
    chain_index = topology.atom(atom_index).residue.index
    if (atom_index_to_monomer_index[atom_index] - 1) not in monomer_index_to_chain_index.keys():
        monomer_index_to_chain_index[(atom_index_to_monomer_index[atom_index] - 1)] = chain_index    #start from 0

monomer_count = len(monomer_index_to_chain_index.keys())
print((monomer_count))

chain_index_to_monomer_indices = {}
for chain_index in range(topology.n_residues):
    chain_index_to_monomer_indices[chain_index] = [key for key,val in monomer_index_to_chain_index.items() if val == chain_index]

all_ij_list = []
chain_lengths = []
for chain_index in range(topology.n_residues):
    monomer_indices = chain_index_to_monomer_indices[chain_index]
    chain_length = len(monomer_indices)
    chain_lengths.append(chain_length)
    ij_list = [i for i in combinations(monomer_indices,2) if 1<=max(i)-min(i) and max(i)-min(i)<=chain_length/2]
    all_ij_list.extend(ij_list)
all_d_list = [max(i)-min(i) for i in all_ij_list]
max_chain_length = max(chain_lengths)

analysis_file = 'extend_back_com.lammpstrj'
lammpstrj_file = open(analysis_file,'r')
frame_count = int(int(os.popen('wc -l {}'.format(analysis_file)).read().split()[0])/(monomer_count+9))

slopes = []
for frame_no in range(frame_count):
    ttt = time.time()
    delta_r = {}
    for i in [i for i in range(1,int(max_chain_length/2)+1) if i<= max_chain_length/2]:
        delta_r[i] = []
        
    for i in range(6):
        line = lammpstrj_file.readline()
    L = float(line.split()[1])
    for i in range(3):
        lammpstrj_file.readline()

    frame_coords = np.zeros((monomer_count,3))
    for monomer_index in range(monomer_count):
        line = lammpstrj_file.readline()
        frame_coords[monomer_index] = np.fromstring(line, sep=' ')[2:]
    frame_coords = wrap_coords(frame_coords,L)

    for ij in all_ij_list:
        delta_r[max(ij)-min(ij)].append(dist_pbc(frame_coords[ij[0]], frame_coords[ij[1]], L)**2)

    delta_r_means = {}
    for i in delta_r.keys():
        delta_r_means[i] = np.mean(delta_r[i])

    
    lnrd = np.array([np.log(delta_r_means[i]) for i in sorted(delta_r.keys())[1:-1]])
    lnd = np.array([np.log(i) for i in sorted(delta_r.keys())[1:-1]])
    slope = linregress(lnd[np.where(lnd<2.2)],lnrd[np.where(lnd<2.2)])[0]
    # print(ttt-time.time())
    slopes.append(slope)


# slopes = {}
# for i in sorted(delta_r.keys())[1:-1]:
#     slope = linregress([1,2,3], [np.log(delta_r_means[i-1]),np.log(delta_r_means[i]),np.log(delta_r_means[i+1])])[0]
#     if slope < 0.3:
#         slope = i-1
#     slopes[i] = slope


f_out = open('scaling_exponent_extend.out','w')
f_out.write('slopes = {}\n'.format(slopes))

# f_out.write('lnd = {}\n'.format(lnd))
# f_out.write('lnrd = {}\n'.format(lnrd))
