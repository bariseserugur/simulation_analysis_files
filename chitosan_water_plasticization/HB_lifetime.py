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

'''This code analyzes the HB lifetimes for polymer-polymer, polymer-solvent, and solvent-solvent hydrogen bonds. 
The output consists of five numpy saves: 
1) the array of tau_values, 
2) polymer-polymer HB lifetime array, 
3) polymer_acceptor-solvent_donor HB lifetime array, 
4) polymer_donor-solvent_acceptor HB lifetime array, 
5) solvent-solvent HB lifetime array.'''

def run_lifetime(universe,donors,hydrogens,acceptors,CUTOFF_DISTANCE = 3.5,CUTOFF_ANGLE = 150,TAU_MAX = 250,WINDOW_STEP = 50):
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

    #Analyze HB Lifetimes for all H-bonds
    tau_frames, hbond_lifetime = HB.lifetime(
        tau_max=TAU_MAX,
        window_step=WINDOW_STEP)
    
    return np.array(tau_frames),np.array(hbond_lifetime)

solvent_HB = {'water':['62','63'],'glycerol':['62','63','64','65','66','67']}

SOLVENT = str(sys.argv[1])
solvent_HB_types = solvent_HB[SOLVENT]

#Output File
f_out = open('HB_lifetimes.out','wb')

#Obtain H-bond donors and acceptors
f = open('HB_300_dynamics.out','rb')

acceptors = np.load(f,allow_pickle=True)
hydrogendonors_unique = np.load(f,allow_pickle=True)

u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs

#Polymer
p_hydrogens = ' '.join(["or name {}".format(i[0]) for i in hydrogendonors_unique if i[0] not in solvent_HB_types]).replace('or ','',1)
p_donors = ' '.join(["or name {}".format(i[1]) for i in hydrogendonors_unique if i[1] not in solvent_HB_types]).replace('or ','',1)
p_acceptors = ' '.join(["or name {}".format(i) for i in acceptors if i not in solvent_HB_types]).replace('or ','',1)

#Solvent
s_hydrogens = ' '.join(["or name {}".format(i[0]) for i in hydrogendonors_unique if i[0] in solvent_HB_types]).replace('or ','',1)
s_donors = ' '.join(["or name {}".format(i[1]) for i in hydrogendonors_unique if i[1] in solvent_HB_types]).replace('or ','',1)
s_acceptors = ' '.join(["or name {}".format(i) for i in acceptors if i in solvent_HB_types]).replace('or ','',1)

#Polymer-Polymer HB Lifetimes
tau_frames,HB_lifetimes = run_lifetime(u,p_donors,p_hydrogens,p_acceptors)
pickle.dump(np.array(tau_frames), f_out) #load list of taus once
pickle.dump(np.array(HB_lifetimes), f_out)

#Polymer_acceptor-Solvent_donor
tau_frames,HB_lifetimes = run_lifetime(u,s_donors,s_hydrogens,p_acceptors)
pickle.dump(np.array(HB_lifetimes), f_out)

#Polymer_donor-Solvent_acceptor
tau_frames,HB_lifetimes = run_lifetime(u,p_donors,p_hydrogens,s_acceptors)
pickle.dump(np.array(HB_lifetimes), f_out)

#Solvent-Solvent HB Lifetimes
tau_frames,HB_lifetimes = run_lifetime(u,s_donors,s_hydrogens,s_acceptors)
pickle.dump(np.array(HB_lifetimes), f_out)

f_out.close()