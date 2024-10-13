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

'''
This code calculates the number of solvent molecules that connect polymer chains via H-bonds.
'''

solvent_HB = {'water':['62','63'],'glycerol':['62','63','64','65','66','67']}

SOLVENT = str(sys.argv[1])
solvent_HB_types = solvent_HB[SOLVENT]

topology = md.load('sys_evaporated.pdb').topology

#label by monomer
monomer_labels = {}
for i in range(1103):
    if i < 24:
        monomer_labels[i] = 1
        continue
    if i > 1079:
        monomer_labels[i] = 50
        continue
    else:
        monomer_labels[i] = (i-24)//22 + 2

def label_monomer(atom_index):
    if atom_index > 11029:
        monomer_index = 500 + int(str(topology.atom(atom_index).residue)[3:]) #artificially label the solvent molecules so their indices are higher
    else:
        index_in_poly = atom_index % 1103
        poly_no = atom_index // 1103
        monomer_index = monomer_labels[index_in_poly] + 50 * poly_no
    return monomer_index

HB_f = open('HB_300_dynamics.out','rb')
HB_array = pickle.load(HB_f)

# acceptors = pickle.load(HB_f)
# hydrogendonors_unique = pickle.load(HB_f)
#Retrieve the full list of H-bonds
# HB_array = np.vstack([pickle.load(HB_f) for i in acceptors for h in hydrogendonors_unique])
#only display H-bonds between monomers
HB_array = HB_array[(HB_array[:,1]<11030) & (HB_array[:,3]<11030)]
frames = np.arange(0,20000,10)

intra_molecular = 0
inter_molecular = 0
for frame in frames:
    print(frame)
    HB_frame = HB_array[HB_array[:,0]==frame]

    residues = np.zeros((len(HB_frame),2))
    AccDon_array = HB_frame[:,[1,3]]
    for index1,row1 in enumerate(AccDon_array):
        for index2,column in enumerate(row1):
            residues[index1,index2] = label_monomer(int(AccDon_array[index1,index2]))
    residues = np.vstack([sorted(i) for i in residues])
    intra_residues = residues[np.sum(np.abs(np.diff(residues)),axis=1) < 2]
    inter_residues = residues[np.sum(np.abs(np.diff(residues)),axis=1) >= 2]
    intra_molecular += len(intra_residues)
    inter_molecular += len(inter_residues)

intra_molecular = intra_molecular/len(frames)
inter_molecular = inter_molecular/len(frames)

#Output File
f_out = open('monomer_monomer.out','w')
f_out.write('intra_molecular = {}\n'.format(intra_molecular))
f_out.write('inter_molecular = {}\n'.format(inter_molecular))