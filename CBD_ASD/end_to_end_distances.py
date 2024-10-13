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

'''Calculate the decorrelation of end-to-end polymer vectors to evaluate the effect of the equilibration procedure.'''

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.sqrt(np.sum(delta**2))

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

analysis_file = 'back_com.lammpstrj'
topology = md.load('sys.pdb').topology

POLY = str(sys.argv[1]).upper()

atom_index_to_monomer_index = return_monomer(POLY)
monomer_index_to_chain_index = {}

for atom_index in atom_index_to_monomer_index.keys():
    chain_index = topology.atom(atom_index).residue.index
    if (atom_index_to_monomer_index[atom_index] - 1) not in monomer_index_to_chain_index.keys():
        monomer_index_to_chain_index[(atom_index_to_monomer_index[atom_index] - 1)] = chain_index    #start from 0

chains = np.arange(0,max(monomer_index_to_chain_index.values()))
max_indices = []
min_indices = []
for chain in chains:
    max_indices.append(max([key for key,value in monomer_index_to_chain_index.items() if value == chain]))
    min_indices.append(min([key for key,value in monomer_index_to_chain_index.items() if value == chain]))

monomer_count = len(monomer_index_to_chain_index.keys())

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

lammpstrj_file = open(analysis_file,'r')
frame_count = int(int(os.popen('wc -l {}'.format(analysis_file)).read().split()[0])/(monomer_count+9))

autocorrelation = []
for frame_no in range(frame_count)[::10]:
    for i in range(6):
        line = lammpstrj_file.readline()
    L = float(line.split()[1])
    for i in range(3):
        lammpstrj_file.readline()

    frame_coords = np.zeros((monomer_count,3))
    for monomer_index in range(monomer_count):
        line = lammpstrj_file.readline()
        frame_coords[monomer_index] = np.fromstring(line, sep=' ')[2:]

    vectors = np.zeros((len(chains),3))
    for chain in chains:
        end_to_end_vector = frame_coords[max_indices[chain]] - frame_coords[min_indices[chain]]
        vectors[chain] = end_to_end_vector

    if frame_no == 0:
        initial_vectors = copy.deepcopy(vectors)
        initial_magnitudes = (initial_vectors[:,None,:] @ initial_vectors[...,None]).ravel()
    
    dot = (vectors[:,None,:] @ initial_vectors[...,None]).ravel()/initial_magnitudes

    autocorrelation.append(np.mean(dot))

f_out = open('end_to_end_distances.out','w')
f_out.write('end_to_end = {}\n'.format(autocorrelation))
f_out.close()