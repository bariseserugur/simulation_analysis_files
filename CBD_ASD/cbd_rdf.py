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

t = md.load('cbd_monomer_com.lammpstrj',top='cbd_monomer_com.pdb')
topology = t.topology

total_atom_count = topology.n_atoms
type_to_number = {'O':1,'tail':2,'hydrophilic':3,'compatibilizer':4}

for group_type in list(type_to_number.keys())[1:]:
    t = md.load('cbd_monomer_com.lammpstrj',top='cbd_monomer_com.pdb')
    topology = t.topology
    atomtype1 = topology.select('name "1"')
    atomtype2 = topology.select('name "{}"'.format(type_to_number[group_type]))

    pair_list = np.array([x for x in list(itertools.product(atomtype1, atomtype2))])

    frame_ct = 0
    g_r_cumulative = np.zeros(200)

    for frame in t:
        r,g_r = md.compute_rdf(frame,pairs=np.array(pair_list),r_range=(0.0,2.0),bin_width=0.01)
        g_r_cumulative+=g_r
        frame_ct += 1

    g_r_cumulative = g_r_cumulative/frame_ct
    f_out = open('monomer_{}_rdf.out'.format(group_type),'w')
    f_out.write('rdf = {}\n'.format(list(g_r_cumulative)))
    f_out.close()


t = md.load('cbd_monomer_com.lammpstrj',top='cbd_monomer_com.pdb')
topology = t.topology
atomtype1 = topology.select('name "3"')
atomtype2 = topology.select('name "3"')

pair_list = np.array([x for x in list(itertools.product(atomtype1, atomtype2)) if topology.atom(x[0])!=topology.atom(x[1])])

frame_ct = 0
g_r_cumulative = np.zeros(200)

for frame in t:
    r,g_r = md.compute_rdf(frame,pairs=np.array(pair_list),r_range=(0.0,2.0),bin_width=0.01)
    g_r_cumulative+=g_r
    frame_ct += 1

g_r_cumulative = g_r_cumulative/frame_ct
f_out = open('cbd_cbd_rdf.out','w')
f_out.write('rdf = {}\n'.format(list(g_r_cumulative)))
f_out.close()
