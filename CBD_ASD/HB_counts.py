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

'''Categorize all analyzed hydrogen bonds and calculate the average number of hydrogen bonds per frame for each category.'''

topology = md.load('sys.pdb').topology
total_atom_count = topology.n_atoms

if topology.atom(total_atom_count-1).residue.n_atoms != 53:
    last_CBD_index = total_atom_count #atom with this index doesn't exist, this deals with the scenarios for 100% polymer systems
    # raise Exception("This ain't CBD!")
else:
    CBD_residue_name = topology.atom(total_atom_count-1).residue.name
    last_CBD_index = min([i.index for i in topology.atoms if i.residue.name == CBD_residue_name])
    # print(len([i for i in topology.residues if i.name==CBD_residue_name]))

f = open('HB_310.out', 'rb')
HB_array = pickle.load(f)
# HB_array_mols = HB_array[:,[1,3]]

frames = set(HB_array[:,0])

ppHB = HB_array[(HB_array[:,1] < last_CBD_index) & (HB_array[:,3] < last_CBD_index)]

#cbds are in third index column
pc_donoracceptor = HB_array[(HB_array[:,1] < last_CBD_index) & (HB_array[:,3] >= last_CBD_index)]

cp_donoracceptor = HB_array[(HB_array[:,1] >= last_CBD_index) & (HB_array[:,3] < last_CBD_index)]
cp_donoracceptor[:,[1,3]] = cp_donoracceptor[:,[3,1]]

pcHB = np.vstack([pc_donoracceptor,cp_donoracceptor])
ccHB = HB_array[(HB_array[:,1] >= last_CBD_index) & (HB_array[:,3] >= last_CBD_index)]

cbd_bridges = 0
for frame in frames:
    frame_pcHB = pcHB[pcHB[:,0]==frame]
    unique, counts = np.unique(frame_pcHB[:,3], return_counts=True)
    multi_cbds = unique[counts>1]
    cbd_bridges += len(multi_cbds)
cbd_bridges/=len(frames)

poly_poly = len(ppHB)/len(frames)
poly_cbd = len(pcHB)/len(frames)
cbd_cbd = len(ccHB)/len(frames)

f_out = open('HB_counts.out','w')
f_out.write('poly_poly = {}\n'.format(poly_poly))
f_out.write('poly_cbd = {}\n'.format(poly_cbd))
f_out.write('cbd_cbd = {}\n'.format(cbd_cbd))
f_out.write('cbd_bridges = {}\n'.format(cbd_bridges))