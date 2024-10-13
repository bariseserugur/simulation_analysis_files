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
import copy

'''This code analyzes the average number of new monomer-monomer H-bonds formed each frame, indicating mm-contact frequency.'''

def run_HB_analysis(universe,donors,hydrogens,acceptors,CUTOFF_DISTANCE = 3.5,CUTOFF_ANGLE = 150):
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
    step=None,
    verbose=True)

    HB_results = HB.results.hbonds

    return HB_results

solvent_HB = {'water':['62','63'],'glycerol':['62','63','64','65','66','67']}

SOLVENT = str(sys.argv[1])
solvent_HB_types = solvent_HB[SOLVENT]

#Obtain H-bond donors and acceptors
f = open('HB_300_dynamics.out','rb')

acceptors = np.load(f,allow_pickle=True)
hydrogendonors_unique = np.load(f,allow_pickle=True)

u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs

#Polymer
p_hydrogens = ' '.join(["or name {}".format(i[0]) for i in hydrogendonors_unique if i[0] not in solvent_HB_types]).replace('or ','',1)
p_donors = ' '.join(["or name {}".format(i[1]) for i in hydrogendonors_unique if i[1] not in solvent_HB_types]).replace('or ','',1)
p_acceptors = ' '.join(["or name {}".format(i) for i in acceptors if i not in solvent_HB_types]).replace('or ','',1)

HB_array = run_HB_analysis(u,p_donors,p_hydrogens,p_acceptors)

frames = np.arange(0,5000,1)

totalcontact = 0
totalframes = 0
for frame in frames:
    HB_frame = HB_array[HB_array[:,0]==frame]
    mm_contacts = [tuple(i) for i in HB_frame[:,[1,2,3]]]
    if frame == 0:
        previous_mm_contacts = copy.deepcopy(mm_contacts)
        continue
    totalcontact += len(list(set(mm_contacts) - set(previous_mm_contacts)))
    previous_mm_contacts = copy.deepcopy(mm_contacts)
    totalframes += 1

f_out = open('monomer_monomer_contact_frequency.out','w')
f_out.write('monomer_monomer_contact_frequency = {}\n'.format(totalcontact/totalframes))