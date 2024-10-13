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

'''
This code calculates the average orientation between the solvent molecules and the monomer atom it forms a hydrogen bond with.
'''

SOLVENT = str(sys.argv[1])
if SOLVENT == 'glycerol':
    raise Exception('It doesnt handle glycerol yet!')

u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs
L = u.trajectory[0].triclinic_dimensions[0][0]
topology = md.load('sys_evaporated.pdb').topology

HB_f = open('HB_300_dynamics.out','rb')
acceptors = pickle.load(HB_f)
hydrogendonors_unique = pickle.load(HB_f)
#Retrieve the full list of H-bonds
HB_array = np.vstack([pickle.load(HB_f) for i in acceptors for h in hydrogendonors_unique])

#solvent-solvent H-bonds
ssHB_array = HB_array[(HB_array[:,1]>11029) & (HB_array[:,3]>11029)]

#only display H-bonds between polymer and solvent molecules
HB_array1 = HB_array[(HB_array[:,1]>11029) & (HB_array[:,3]<11030)]
HB_array2 = HB_array[(HB_array[:,1]<11030) & (HB_array[:,3]>11029)]
HB_array = np.vstack([HB_array1,HB_array2])
frames = np.arange(0,5000,100)

#indices of oxygens that connect the sugar rings
connection_oxygens = np.array([22, 44, 66, 88, 110, 132, 154, 176, 198, 220, 242, 264, 286, 308, 330, 352, 374, 396, 418, 440, 462, 484, 506, 528, 550, 572, 594, 616, 638, 660, 682, 704, 726, 748, 770, 792, 814, 836, 858, 880, 902, 924, 946, 968, 990, 1012, 1034, 1056, 1078, 1100, 1125, 1147, 1169, 1191, 1213, 1235, 1257, 1279, 1301, 1323, 1345, 1367, 1389, 1411, 1433, 1455, 1477, 1499, 1521, 1543, 1565, 1587, 1609, 1631, 1653, 1675, 1697, 1719, 1741, 1763, 1785, 1807, 1829, 1851, 1873, 1895, 1917, 1939, 1961, 1983, 2005, 2027, 2049, 2071, 2093, 2115, 2137, 2159, 2181, 2203, 2228, 2250, 2272, 2294, 2316, 2338, 2360, 2382, 2404, 2426, 2448, 2470, 2492, 2514, 2536, 2558, 2580, 2602, 2624, 2646, 2668, 2690, 2712, 2734, 2756, 2778, 2800, 2822, 2844, 2866, 2888, 2910, 2932, 2954, 2976, 2998, 3020, 3042, 3064, 3086, 3108, 3130, 3152, 3174, 3196, 3218, 3240, 3262, 3284, 3306, 3331, 3353, 3375, 3397, 3419, 3441, 3463, 3485, 3507, 3529, 3551, 3573, 3595, 3617, 3639, 3661, 3683, 3705, 3727, 3749, 3771, 3793, 3815, 3837, 3859, 3881, 3903, 3925, 3947, 3969, 3991, 4013, 4035, 4057, 4079, 4101, 4123, 4145, 4167, 4189, 4211, 4233, 4255, 4277, 4299, 4321, 4343, 4365, 4387, 4409, 4434, 4456, 4478, 4500, 4522, 4544, 4566, 4588, 4610, 4632, 4654, 4676, 4698, 4720, 4742, 4764, 4786, 4808, 4830, 4852, 4874, 4896, 4918, 4940, 4962, 4984, 5006, 5028, 5050, 5072, 5094, 5116, 5138, 5160, 5182, 5204, 5226, 5248, 5270, 5292, 5314, 5336, 5358, 5380, 5402, 5424, 5446, 5468, 5490, 5512, 5537, 5559, 5581, 5603, 5625, 5647, 5669, 5691, 5713, 5735, 5757, 5779, 5801, 5823, 5845, 5867, 5889, 5911, 5933, 5955, 5977, 5999, 6021, 6043, 6065, 6087, 6109, 6131, 6153, 6175, 6197, 6219, 6241, 6263, 6285, 6307, 6329, 6351, 6373, 6395, 6417, 6439, 6461, 6483, 6505, 6527, 6549, 6571, 6593, 6615, 6640, 6662, 6684, 6706, 6728, 6750, 6772, 6794, 6816, 6838, 6860, 6882, 6904, 6926, 6948, 6970, 6992, 7014, 7036, 7058, 7080, 7102, 7124, 7146, 7168, 7190, 7212, 7234, 7256, 7278, 7300, 7322, 7344, 7366, 7388, 7410, 7432, 7454, 7476, 7498, 7520, 7542, 7564, 7586, 7608, 7630, 7652, 7674, 7696, 7718, 7743, 7765, 7787, 7809, 7831, 7853, 7875, 7897, 7919, 7941, 7963, 7985, 8007, 8029, 8051, 8073, 8095, 8117, 8139, 8161, 8183, 8205, 8227, 8249, 8271, 8293, 8315, 8337, 8359, 8381, 8403, 8425, 8447, 8469, 8491, 8513, 8535, 8557, 8579, 8601, 8623, 8645, 8667, 8689, 8711, 8733, 8755, 8777, 8799, 8821, 8846, 8868, 8890, 8912, 8934, 8956, 8978, 9000, 9022, 9044, 9066, 9088, 9110, 9132, 9154, 9176, 9198, 9220, 9242, 9264, 9286, 9308, 9330, 9352, 9374, 9396, 9418, 9440, 9462, 9484, 9506, 9528, 9550, 9572, 9594, 9616, 9638, 9660, 9682, 9704, 9726, 9748, 9770, 9792, 9814, 9836, 9858, 9880, 9902, 9924, 9949, 9971, 9993, 10015, 10037, 10059, 10081, 10103, 10125, 10147, 10169, 10191, 10213, 10235, 10257, 10279, 10301, 10323, 10345, 10367, 10389, 10411, 10433, 10455, 10477, 10499, 10521, 10543, 10565, 10587, 10609, 10631, 10653, 10675, 10697, 10719, 10741, 10763, 10785, 10807, 10829, 10851, 10873, 10895, 10917, 10939, 10961, 10983, 11005, 11027])

HB_orientations = []
ON = []
PPratios = [[] for i in range(50)] #add 1 if this solvent is connected to another polymer 0 if it is not
for frameix, frame in enumerate(frames):
    HB_frame = HB_array[HB_array[:,0]==frame][:,[1,3]] #only acceptor and donor
    HB_frame = np.vstack([sorted(i) for i in HB_frame])
    ssHB_frame = ssHB_array[ssHB_array[:,0]==frame][:,[1,3]]

    frame_xyz = u.trajectory[frameix].positions
    for row in HB_frame:
        solvent_index = int(row[1])
        polymer_atom_index = int(row[0])
        solvent_residue = topology.atom(solvent_index).residue.index
        solvent_atom_ix = np.array([i.index for i in topology.residue(solvent_residue).atoms])
        solvent_director = (frame_xyz[solvent_atom_ix[1]]-frame_xyz[solvent_atom_ix[0]]) + (frame_xyz[solvent_atom_ix[2]]-frame_xyz[solvent_atom_ix[0]])
        # solvent_polymer_director = frame_xyz[solvent_atom_ix[0]] - frame_xyz[int(row[0])]
        connect_O1, connect_O2 = connection_oxygens[np.argsort(np.abs(connection_oxygens-polymer_atom_index))[0]],connection_oxygens[np.argsort(np.abs(connection_oxygens-polymer_atom_index))[1]]
        solvent_polymer_director = frame_xyz[connect_O1] - frame_xyz[connect_O2]
        cos = np.abs(np.dot(solvent_director,solvent_polymer_director)/norm(solvent_director)/norm(solvent_polymer_director))
        HB_orientations.append(cos)

        categories = np.linspace(0,1,51)
        cat = [categories[i]<cos and categories[i+1]>=cos for i in range(0, len(categories)-1)].index(True)

        #Check if this solvent is connected to another polymer
        HBlist = HB_frame[HB_frame[:,1]==solvent_index][:,0]
        if max(HBlist) - min(HBlist) > 20 and len(HBlist)>1 and solvent_index not in ssHB_frame:
            PPratios[cat].append(1)
        else:
            PPratios[cat].append(0)

        if topology.atom(int(row[0])).element.symbol == 'N':
            ON.append(1)
        elif topology.atom(int(row[0])).element.symbol == 'O':
            ON.append(0)
        else:
            print('neither!!!')

average_orientation = np.mean(HB_orientations)
freqs,vals = np.histogram(HB_orientations,bins=np.linspace(0,1,51))
freqs = list(freqs)
vals = list(vals)
PPratios = [np.mean(i) for i in PPratios]

meanON = np.mean(ON)

#Output File
f_out = open('HB_orientation.out','w')
f_out.write('average_orientation = {}\n'.format(average_orientation))
f_out.write('freqs = {}\n'.format(freqs))
f_out.write('vals = {}\n'.format(vals))
f_out.write('PPratios = {} #ratio of polymer-bonded solvents that form HB with another polymer \n'.format(PPratios))
f_out.write('meanON = {} #0 is Oxygen, 1 is Nitrogen H-bonds \n'.format(meanON))