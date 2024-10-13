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

sys.setrecursionlimit(1000000)

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
p_donors = ''
w_donors = ''
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
    AccDon_array = np.vstack([sorted(i) for i in AccDon_array])
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

        solvent_HBs1 = AccDon_array[solresidues[:,0]==solvent_mol][:,1]
        solvent_HBs2 = AccDon_array[solresidues[:,1]==solvent_mol][:,0]
        sss1 = AccDon_array[solresidues[:,0]==solvent_mol][:,0]
        sss2 = AccDon_array[solresidues[:,1]==solvent_mol][:,1]
        sss = np.hstack([sss1,sss2])
        sss = sss[0]
        solvent_HBss = np.hstack([solvent_HBs1,solvent_HBs2])
        solvent_HBss = np.unique(solvent_HBss)

        # print(connect_points,solvent_HBs)
        
        if len(solvent_HBs) == 1:
            zeros += 1
        elif any(solvent_HBs>501):
            # print(solvent_HBss)
            solvents += 1
            if any(solvent_HBs<501):
                for i in solvent_HBss:
                    if i < 11030:
                        p_donors += " or index {}".format(int(i))
                w_donors += " or index {}".format(int(sss))
            
        elif max(solvent_HBs) - min(solvent_HBs) > 1:
            polymers += 1
            # for i in solvent_HBss:
            #     p_donors += " or index {}".format(int(i))
            # w_donors += " or index {}".format(int(sss))
        else:
            zeros += 1
    break

p_donors = p_donors.replace('or ','',1)[1:]
w_donors = w_donors.replace('or ','',1)[1:]

solvent_HB = {'water':['62','63'],'glycerol':['62','63','64','65','66','67']}

SOLVENT = str(sys.argv[1])
solvent_HB_types = solvent_HB[SOLVENT]

#Output File
f_out = open('pwplifetimes.out','wb')

#Obtain H-bond donors and acceptors
f = open('HB_300_dynamics.out','rb')

acceptors = np.load(f,allow_pickle=True)
hydrogendonors_unique = np.load(f,allow_pickle=True)

u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs

tau_frames,HB_lifetimes = run_lifetime(u,w_donors,'element H',p_donors)
pickle.dump(np.array(tau_frames), f_out) #load list of taus once
pickle.dump(np.array(HB_lifetimes), f_out)

f_out.close()