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

It also calculates the number of short/long connections (tying monomers with a distance of 10 monomers), and the number of HB formed by a solvent that is currently H-bonded to a monomer.
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
acceptors = pickle.load(HB_f)
hydrogendonors_unique = pickle.load(HB_f)
#Retrieve the full list of H-bonds
HB_array = np.vstack([pickle.load(HB_f) for i in acceptors for h in hydrogendonors_unique])

#List of HBonds that include at least one solvent
solvent_HB_array = HB_array[(HB_array[:,1]>11029) | (HB_array[:,3]>11029)]

#only display H-bonds between polymer and solvent molecules
HB_array1 = HB_array[(HB_array[:,1]>11029) & (HB_array[:,3]<11030)]
HB_array2 = HB_array[(HB_array[:,1]<11030) & (HB_array[:,3]>11029)]
HB_array = np.vstack([HB_array1,HB_array2])

frames = np.arange(0,5000,100)

connection_count = 0
monomer_gap = []
solvent_HB_counts =[]
zeros = 0
solvents = 0
polymers = 0
for frame in frames:
    print(frame)
    HB_frame = HB_array[HB_array[:,0]==frame]
    solvent_HB_frame = solvent_HB_array[solvent_HB_array[:,0]==frame]

    residues = np.zeros((len(HB_frame),2))
    AccDon_array = HB_frame[:,[1,3]]
    for index1,row1 in enumerate(AccDon_array):
        for index2,column in enumerate(row1):
            residues[index1,index2] = label_monomer(int(AccDon_array[index1,index2]))
    residues = np.vstack([sorted(i) for i in residues])

    solresidues = np.zeros((len(solvent_HB_frame),2))
    AccDon_array = solvent_HB_frame[:,[1,3]]
    for index1,row1 in enumerate(AccDon_array):
        for index2,column in enumerate(row1):
            solresidues[index1,index2] = label_monomer(int(AccDon_array[index1,index2]))
    solresidues = np.vstack([sorted(i) for i in solresidues])

    for solvent_mol in set(residues[:,1]):
        connect_points = residues[residues[:,1]==solvent_mol][:,0]
        solvent_HBs1 = solresidues[solresidues[:,0]==solvent_mol][:,1]
        solvent_HBs2 = solresidues[solresidues[:,1]==solvent_mol][:,0]
        solvent_HBs = np.hstack([solvent_HBs1,solvent_HBs2])
        solvent_HBs = np.unique(solvent_HBs)
        # print(connect_points,solvent_HBs)
        
        if len(solvent_HBs) == 1:
            zeros += 1
        elif any(solvent_HBs>501):
            solvents += 1
        elif max(solvent_HBs) - min(solvent_HBs) > 1:
            polymers += 1
        else:
            zeros += 1

        unique_connect_points = set(connect_points)
        solvent_HB_counts.append(len(solvent_HBs))

        if len(unique_connect_points) > 1:
            if max(unique_connect_points) - min(unique_connect_points) > 1:
                connection_count += 1

                if np.mean(np.abs(np.diff(np.array(list(unique_connect_points))))) >= 10:
                    # monomer_gap += 1
                    monomer_gap.append(1)
                else:
                    monomer_gap.append(0)
                # monomer_gap.append(np.mean(np.abs(np.diff(np.array(list(unique_connect_points))))))

connection_count = connection_count/len(frames)
monomer_gap = np.mean(monomer_gap)
solvent_HB_counts = np.mean(solvent_HB_counts) #total number of H-bonds formed by a solvent molecule that is currently forming a H-bond with a monomer
zeros = zeros/len(frames)
solvents = solvents/len(frames)
polymers = polymers/len(frames)

#Output File
f_out = open('connection_counts.out','w')
f_out.write('connection_count = {}\n'.format(connection_count))
f_out.write('monomer_gap = {}\n'.format(monomer_gap))
f_out.write('solvent_HB_counts = {}\n'.format(solvent_HB_counts))
f_out.write('zeros = {}\n'.format(zeros))
f_out.write('solvents = {}\n'.format(solvents))
f_out.write('polymers = {}\n'.format(polymers))