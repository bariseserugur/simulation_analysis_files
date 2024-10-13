import numpy as np
import os
import mdtraj as md
import pickle
import time

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

#Filter out any non-solvent H-bonds
HB_solvent = HB_array[np.isin(HB_array[:,1],solvents)] #solvent donors
HB_solvent = HB_solvent[np.isin(HB_solvent[:,3],solvents)] #solvent acceptors


cluster_counts = []
mean_cluster_sizes = []
reached_depths = []
for frame in frames:
    print(frame)
    HB_frame = HB_solvent[HB_solvent[:,0]==frame] #H-bonds that belong to the given frame

    #if no solvent-solvent H-bonds in that frame, add 0 and continue
    if len(HB_frame) == 0:
        cluster_counts.append(0)
        mean_cluster_sizes.append(0)
        continue

    #Convert H-bond donor/acceptor indices to residue numbers, then filter out such that frame number, distance etc. are not displayed (yields an (N,2) array)
    residues = np.zeros((len(HB_frame),2))
    AccDon_array = HB_frame[:,[1,3]].astype('int')
    for index1,row1 in enumerate(AccDon_array):
        for index2,column in enumerate(row1):
            residues[index1,index2] = topology.atom(AccDon_array[index1,index2]).residue.index

    sortedresidues = np.array([sorted(i) for i in residues])
    #filter out multiple H-bonds between same pair
    residues = np.unique(sortedresidues, axis=0)
    residues = residues[residues[:, 0].argsort()]

    groups = []
    max_length = len(residues)
    for i in range(max_length): #you can only have as many groups as the number of residues (i.e. each group is one molecule)
        max_depth = 1000
        if len(residues) == 0:
            break
        connections = {}
        others = {}
        for i in range(max_depth):
            others[i] = []
        
        first_pair = residues[0]
        r1 = first_pair[0]
        r2 = first_pair[1]
        group = [r1]
        connections[0] = np.where(residues==r1)

        #convert H-bonds that include residue 1 (r1) to their residue numbers
        others[0] = [int(residues[x[0]][x[1]]) for x in [[connections[0][0][i],np.abs(1-connections[0][1][i])] for i in range(len(connections[0][0]))]]
        group.extend(others[0])

        #filter out H-bonds that have been analyzed (i.e. ones that include residue 1)
        residues = np.array([i for i in residues if r1 not in i])
        depth = 0
        max_depth_reached = 0
        while max_depth > depth:
            for r1 in others[depth]:
                connections[depth+1] = np.where(residues==r1)
                others_grp = [residues[x[0]][x[1]] for x in [[connections[depth+1][0][i],np.abs(1-connections[depth+1][1][i])] for i in range(len(connections[depth+1][0]))]]
                others[depth+1].extend(others_grp)
                group.extend(others_grp)
                residues = np.array([i for i in residues if r1 not in i])
            if len(others[depth+1]) == 0:
                depth = max_depth
                max_depth_reached = 1
            depth+=1
        groups.append(group)
    distribution = [len(i) for i in groups]
    print(distribution)
    print(abc)
    cluster_counts.append(len(groups))
    mean_cluster_sizes.append(np.mean([len(i) for i in groups]))
    reached_depths.append(max_depth_reached)

    # freqs,vals = np.histogram(distribution,bins=list(np.arange(0,30+2,1))) 
    # print('freqs = {}\n'.format(list(freqs)),'vals = {}'.format(list(vals)))
    # print(abc)

# f_out = open('solvent_clusters.out','w')
# f_out.write('cluster_count = {}\n'.format(np.mean(cluster_counts)))
# f_out.write('cluster_size = {}\n'.format(np.mean(mean_cluster_sizes)))
# f_out.write('max_reached_depths = {}\n'.format(reached_depths))
# f_out.close()
