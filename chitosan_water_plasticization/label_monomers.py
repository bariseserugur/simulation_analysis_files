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

def label_monomers():
    topology = md.load('sys_evaporated.pdb').topology
    
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
    
    index_to_monomer = {}
    for atom_index in range(11030): 
        index_in_poly = atom_index % 1103
        poly_no = atom_index // 1103
        monomer_index = monomer_labels[index_in_poly] + 50 * poly_no
        index_to_monomer[atom_index] = monomer_index
    return index_to_monomer
