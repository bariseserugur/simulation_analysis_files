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
from label_monomers import label_monomers

'''Analyze the number of polymer-water-polymer bridges (per volume).'''
index_to_monomer = label_monomers()

f = open('HB_300_dynamics.out', 'rb')
HB_array = pickle.load(f)

frames_to_analyze = np.arange(0,int(max(HB_array[:,0])),100)

#water-water H-bond
water_water = HB_array[(HB_array[:,1]>11029) & (HB_array[:,3]>11029)]

#water donor, polymer acceptor
polymer_solvent = HB_array[(HB_array[:,1]>11029) & (HB_array[:,3]<11030)]
#water acceptor, but switch the columns 1 and 3
water_acceptor = HB_array[(HB_array[:,1]<11030) & (HB_array[:,3]>11029)]
water_acceptor[:, [1, 3]] = water_acceptor[:, [3, 1]]

#water in column #1, polymer in column #3
polymer_solvent = np.vstack([polymer_solvent,water_acceptor])

topology = md.load('sys_evaporated.pdb').topology

#all water indices are replaced with water molecule indices
polymer_solvent[:,1] = np.array([topology.atom(int(i)).residue.index for i in polymer_solvent[:,1]])
#same with water-water interactions
water_water[:,1] = np.array([topology.atom(int(i)).residue.index for i in water_water[:,1]])
water_water[:,3] = np.array([topology.atom(int(i)).residue.index for i in water_water[:,3]])

#all polymer indices are replaced with monomer indices
polymer_solvent[:,3] = np.array([index_to_monomer[i] for i in polymer_solvent[:,3]])

dry_bridge = 0
wet_bridge = 0
no_bridge = 0
for frame_index in frames_to_analyze:
    frame_polymer_solvent = polymer_solvent[polymer_solvent[:,0]==frame_index]
    frame_water_water = water_water[water_water[:,0]==frame_index]

    unique_waters = np.unique(frame_polymer_solvent[:,1])
    for water in unique_waters:
        that_water_bonds = frame_polymer_solvent[frame_polymer_solvent[:,1]==water]
        if np.ptp(that_water_bonds[:,3]) > 1:
            if water in frame_water_water:
                wet_bridge += 1
            else:
                dry_bridge += 1
        else:
            no_bridge += 1

wet_bridge /= len(frames_to_analyze)
dry_bridge /= len(frames_to_analyze)
no_bridge /= len(frames_to_analyze)

f_out = open('bridge_counts.out','w')
f_out.write('wet_bridge = {}\n'.format(wet_bridge))
f_out.write('dry_bridge = {}\n'.format(dry_bridge))
f_out.write('no_bridge = {}\n'.format(no_bridge))
f_out.close()