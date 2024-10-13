import numpy as np
import joblib
from joblib import Parallel,delayed
import argparse
import sys
import time
import os
import pickle
import mdtraj as md
import MDAnalysis as mda

'''
This code calculates the average number of exchanges of the solvent molecules neighboring each polymer segment's O and N 
for each frame. It returns the average number of exchanges per frame, average number of neighboring solvent molecules,
and the number of exchanges divided/normalized by the number of neighboring solvent molecules.

input parameters is the proximity distance (default 3 Angstroms) and the total length of simulation (default 100 ps)
'''

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

# def find_proximity(segment_coords, solvent_coords, L, prox):
#     # calculate distance from polymer molecules
#     # if distance is < prox Angstroms: within proximity
#     neigh_solvents = set([])
#     for segment_coord in segment_coords:
#         prox_solvents = np.where(dist_pbc(solvent_coords, segment_coord, L)**0.5 < prox)
#         solvents_to_add = solvent_oxygens[prox_solvents]
#         for i in solvents_to_add:
#             neigh_solvents.add(i)
#     return list(neigh_solvents)

def find_proximity(all_polymer_segments,u,frame_list, L, prox):
    neigh_every_frame = []
    for frame_no in frame_list:
        all_coords = u.trajectory[frame_no].positions
        solvent_coords = all_coords[solvent_oxygens]
        neighs_one_frame = []
        for i in range(len(all_polymer_segments)):
            segment_coords = all_coords[all_polymer_segments[i]]
            # calculate distance from polymer molecules
            # if distance is < prox Angstroms: within proximity
            neigh_solvents = set([])
            for segment_coord in segment_coords:
                prox_solvents = np.where(dist_pbc(solvent_coords, segment_coord, L)**0.5 < prox)
                solvents_to_add = solvent_oxygens[prox_solvents]
                for i in solvents_to_add:
                    neigh_solvents.add(i)
            neighs_one_frame.append(list(neigh_solvents))
        neigh_every_frame.append(neighs_one_frame)
    return neigh_every_frame

def check_frame_change(frame_indices,all_resultss):
    totaldiff = 0
    framexsegment = 0
    totalsolventneigh = 0
    for frame in frame_indices:
        if frame == 0:
            continue
        for segment_no in range(len(all_polymer_segments)):
            prev_segment = all_resultss[frame-1][segment_no]
            frame_segment = all_resultss[frame][segment_no]
            if len(prev_segment) == 0:
                continue
            totalsolventneigh += len(all_resultss[frame][segment_no])
            totaldiff += len(list(set(prev_segment) - set(frame_segment))) #number of solvent molecules that left the proximity range in this step
            #totaldiff += len(set(prev_segment).symmetric_difference(set(frame_segment))) #difference between two lists
            framexsegment += 1
    return (totaldiff,totalsolventneigh,framexsegment)

u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs
L = u.trajectory[0].triclinic_dimensions[0][0]
solvent_oxygens = u.select_atoms('resname BBB and type O').indices

all_frames = list(range(len(u.trajectory)))
frame_ct = len(u.trajectory)
frames_dist = split(all_frames,50)

# one_polymer_segments = [np.array(i) for i in [range(222),range(222,442),range(442,662),range(662,882),range(882,1103)]]
# all_polymer_segments = []
# for poly_no in range(10):
#     all_polymer_segments.extend([i+poly_no*1103 for i in one_polymer_segments])
# all_atoms = [list(i) for i in all_polymer_segments.copy()]
# all_polymer_segments = []
# #FILTER SEGMENTS SUCH THAT ONLY O and N are present to check neighbors
# for segmentcopy in all_atoms:
#     filtered_segment = [i for i in segmentcopy if t.topology.atom(int(i)).element.symbol in ['O','N']]
#     all_polymer_segments.append(np.array(filtered_segment))
# #

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
        monomer_index = 500 + int(u.atoms[atom_index].residue.resid) #artificially label the solvent molecules so their indices are higher
    else:
        index_in_poly = atom_index % 1103
        poly_no = atom_index // 1103
        monomer_index = monomer_labels[index_in_poly] + 50 * poly_no
    return monomer_index

all_polymer_segments = [[] for i in range(500)]
for i in range(11030):
    if u.atoms[i].element not in ['O','N']:
        continue
    all_polymer_segments[int(label_monomer(i)-1)].append(i)
all_polymer_segments = [np.array(i) for i in all_polymer_segments]

ncores = 50
all_frames = list(range(len(u.trajectory)))
frame_ct = len(u.trajectory)
frames_dist = split(all_frames,ncores)
neigh_every_frame = Parallel(n_jobs=ncores)(delayed(find_proximity)(all_polymer_segments,u,frames_dist[i], L, 3) for i in range(ncores))
all_results = []
for index,obj in enumerate(neigh_every_frame):
    all_results.extend(obj)

# ncores = 50
# solvents_each_frame = []
# tt = time.time()
# for frame in range(len(xyz)):
#     all_coords = xyz[frame]*10
#     close_solvents = Parallel(n_jobs=ncores)(delayed(find_proximity)(range(int(i*10),int((i+1)*10)),all_coords[solvent_oxygens],L,3.5) for i in range(ncores)) #within 3.5 Angstroms
#     cs = close_solvents.copy()
#     close_solvents= []
#     for i in cs:
#         close_solvents+=i
#     print(time.time()-tt)
#     solvents_each_frame.append(close_solvents)
# print(abc)

ncores = 10
frames_dist = split(all_frames,ncores)
outputs = Parallel(n_jobs=10, backend='multiprocessing')(delayed(check_frame_change)(frames_dist[i],all_results) for i in range(ncores))

totaldiff = 0
totalsolventneigh = 0
framexsegment = 0
for i in outputs:
    totaldiff += i[0]
    totalsolventneigh += i[1]
    framexsegment += i[2]
solvent_flux = totaldiff/(framexsegment)
totalsolventneigh = totalsolventneigh/(framexsegment)

f_out = open('polymer_solvent_flux.out','w')
f_out.write('polymer_solvent_flux = {}\n'.format(solvent_flux))
f_out.write('solvent_neigh_count = {}\n'.format(totalsolventneigh))
f_out.write('fluxperneigh = {}\n'.format(solvent_flux/totalsolventneigh))
f_out.close()