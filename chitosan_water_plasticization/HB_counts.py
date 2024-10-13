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
This code calculates the average number of H-bonds between pairs of four categories. 
The output is the average number of:
1) polymer-polymer H-bonds 
2) polymer_acceptor-solvent_donor H-bonds
3) polymer_donor-solvent_acceptor H-bonds
4) solvent-solvent H-bonds
'''

def run_HB(universe,donors,hydrogens,acceptors,CUTOFF_DISTANCE = 3.5,CUTOFF_ANGLE = 150):
    HB = HydrogenBondAnalysis(
    universe=universe,
    donors_sel=donors,
    hydrogens_sel=hydrogens,
    acceptors_sel=acceptors,
    d_a_cutoff=CUTOFF_DISTANCE,
    d_h_a_angle_cutoff=CUTOFF_ANGLE,
    update_selections=False)
 
    HB.run(
    start=None,
    stop=None,
    step=100,
    verbose=True)
    HB_results = HB.results.hbonds 
    return HB_results

solvent_HB = {'water':['62','63'],'glycerol':['62','63','64','65','66','67']}

SOLVENT = str(sys.argv[1])
solvent_HB_types = solvent_HB[SOLVENT]
            
#Output File
f_out = open('HB_counts.out','w')

#Obtain H-bond donors and acceptors
f = open('HB_300_dynamics.out','rb')

acceptors = pickle.load(f)
hydrogendonors_unique = pickle.load(f)

u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs
frame_count = len(u.trajectory)/100 #HB data is collected every 100 steps

#Polymer
p_hydrogens = ' '.join(["or name {}".format(i[0]) for i in hydrogendonors_unique if i[0] not in solvent_HB_types]).replace('or ','',1)
p_donors = ' '.join(["or name {}".format(i[1]) for i in hydrogendonors_unique if i[1] not in solvent_HB_types]).replace('or ','',1)
p_acceptors = ' '.join(["or name {}".format(i) for i in acceptors if i not in solvent_HB_types]).replace('or ','',1)

#Solvent
s_hydrogens = ' '.join(["or name {}".format(i[0]) for i in hydrogendonors_unique if i[0] in solvent_HB_types]).replace('or ','',1)
s_donors = ' '.join(["or name {}".format(i[1]) for i in hydrogendonors_unique if i[1] in solvent_HB_types]).replace('or ','',1)
s_acceptors = ' '.join(["or name {}".format(i) for i in acceptors if i in solvent_HB_types]).replace('or ','',1)

#Polymer-Polymer HB Lifetimes
HB_results = run_HB(u,p_donors,p_hydrogens,p_acceptors)
f_out.write('pp = {}\n'.format(len(HB_results)/frame_count))

#Polymer_acceptor-Solvent_donor
HB_results = run_HB(u,s_donors,s_hydrogens,p_acceptors)
f_out.write('pasd = {}\n'.format(len(HB_results)/frame_count))

#Polymer_donor-Solvent_acceptor
HB_results = run_HB(u,p_donors,p_hydrogens,s_acceptors)
f_out.write('pdsa = {}\n'.format(len(HB_results)/frame_count))

#Solvent-Solvent HB Lifetimes
HB_results = run_HB(u,s_donors,s_hydrogens,s_acceptors)
f_out.write('ss = {}\n'.format(len(HB_results)/frame_count))

f_out.close()