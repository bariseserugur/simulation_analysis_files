import numpy as np
import os
import mdtraj as md
import pickle
import time
import copy

'''This script calculates the number and size of solvent clusters that are connected with 
at least one H-bond'''
frames = np.array([0]) #np.arange(0,200,1000)

HB_array = pickle.load(open('HB_300_dynamics.out','rb'))
HB_array = HB_array[np.isin(HB_array[:,0],frames)]

# acceptors = pickle.load(HB_f)
# hydrogendonors_unique = pickle.load(HB_f)

# #Retrieve the full list of H-bonds
# HB_array = np.vstack([pickle.load(HB_f) for i in acceptors for h in hydrogendonors_unique])

#Obtain list of indices of solvent atoms
topology = md.load('sys_evaporated.pdb').topology
solvents = topology.select('resname BBB')
min_water_index = min([topology.atom(i).residue.index for i in solvents])

#Filter out any non-solvent H-bonds
HB_solvent = HB_array[np.isin(HB_array[:,1],solvents)] #solvent donors
HB_solvent = HB_solvent[np.isin(HB_solvent[:,3],solvents)] #solvent acceptors


cluster_counts = []
mean_cluster_sizes = []

for frame in frames:
    HB_frame = HB_solvent[HB_solvent[:,0]==frame] #H-bonds that belong to the given frame

    #if no solvent-solvent H-bonds in that frame, add 0 and continue
    if len(HB_frame) == 0:
        cluster_counts.append(0)
        mean_cluster_sizes.append(0)
        groups = []
        continue

    #Convert H-bond donor/acceptor indices to residue numbers, then filter out such that frame number, distance etc. are not displayed (yields an (N,2) array)
    residues = np.zeros((len(HB_frame),2))
    AccDon_array = HB_frame[:,[1,3]].astype('int')
    for index1,row1 in enumerate(AccDon_array):
        for index2,column in enumerate(row1):
            residues[index1,index2] = topology.atom(AccDon_array[index1,index2]).residue.index

    #convert such that 0 is water molecule #1
    residues = residues - min_water_index

    sortedresidues = np.array([sorted(i) for i in residues])
    #filter out multiple H-bonds between same pair
    residues = np.unique(sortedresidues, axis=0)
    residues = residues[residues[:, 0].argsort()].astype(int)

    print('start')
    groups = []
    unpaired_residues = copy.deepcopy(residues)
    while len(unpaired_residues) > 0:
        matched_unique_residues = [unpaired_residues[0][0]]
        checked_residues = []
        while len(checked_residues) != len(matched_unique_residues):
            for residue in matched_unique_residues:
                if residue in checked_residues:
                    continue
                mask = np.any(residue==unpaired_residues,axis=1)
                matched_unique_residues = np.unique(np.hstack((matched_unique_residues,np.unique(unpaired_residues[mask]))))
                unpaired_residues = unpaired_residues[mask==False]
                checked_residues.append(residue)
        groups.append(checked_residues)
    
    distribution = [len(i) for i in groups]
    cluster_counts.append(len(groups))
    mean_cluster_sizes.append(np.mean([len(i) for i in groups]))


    # freqs,vals = np.histogram(distribution,bins=list(np.arange(0,30+2,1))) 
    # print('freqs = {}\n'.format(list(freqs)),'vals = {}'.format(list(vals)))
    # print(abc)

f_out = open('solvent_clusters.out','w')
f_out.write('groups = {}\n'.format(groups))
# f_out.write('cluster_count = {}\n'.format(np.mean(cluster_counts)))
# f_out.write('cluster_size = {}\n'.format(np.mean(mean_cluster_sizes)))
# f_out.write('max_reached_depths = {}\n'.format(reached_depths))
f_out.close()
