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
import copy
import joblib
from joblib import Parallel,delayed

'''This code analyzes the HB lifetimes for polymer-polymer, polymer-solvent, and solvent-solvent hydrogen bonds. 
The output consists of five numpy saves: 
1) the array of tau_values, 
2) polymer-polymer HB lifetime array, 
3) polymer_acceptor-solvent_donor HB lifetime array, 
4) polymer_donor-solvent_acceptor HB lifetime array, 
5) solvent-solvent HB lifetime array.'''

sttime = time.time()

solvent_HB = {'water':['62','63'],'glycerol':['62','63','64','65','66','67']}

SOLVENT = str(sys.argv[1])
solvent_HB_types = solvent_HB[SOLVENT]


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def read_positions(frame_range):
    THIS CODE IS WRONG CHECK orientational_relaxayion.py for correct version
    u = mda.Universe('modified_sys_evaporated.pdb','modified_300_dynamics.lammpstrj', format='LAMMPSDUMP', dt=0.01) #data collected every 10 fs
    L = u.trajectory[0].triclinic_dimensions[0][0] #constant L since NVE ensemble
    xyz_list = []
    for frame in frame_range:
        frame_xyz = u.trajectory[frame].positions
        frame_xyz_delta = frame_xyz - frame_xyz[0]
        frame_xyz = np.where(frame_xyz_delta > 0.5*L, frame_xyz-L, np.where(frame_xyz_delta < -0.5*L, frame_xyz+L, frame_xyz))
        xyz_list.append(frame_xyz)
    return xyz_list

topology = md.load('sys_evaporated.pdb').topology

connection_oxygens = np.array([22, 44, 66, 88, 110, 132, 154, 176, 198, 220, 242, 264, 286, 308, 330, 352, 374, 396, 418, 440, 462, 484, 506, 528, 550, 572, 594, 616, 638, 660, 682, 704, 726, 748, 770, 792, 814, 836, 858, 880, 902, 924, 946, 968, 990, 1012, 1034, 1056, 1078, 1100, 1125, 1147, 1169, 1191, 1213, 1235, 1257, 1279, 1301, 1323, 1345, 1367, 1389, 1411, 1433, 1455, 1477, 1499, 1521, 1543, 1565, 1587, 1609, 1631, 1653, 1675, 1697, 1719, 1741, 1763, 1785, 1807, 1829, 1851, 1873, 1895, 1917, 1939, 1961, 1983, 2005, 2027, 2049, 2071, 2093, 2115, 2137, 2159, 2181, 2203, 2228, 2250, 2272, 2294, 2316, 2338, 2360, 2382, 2404, 2426, 2448, 2470, 2492, 2514, 2536, 2558, 2580, 2602, 2624, 2646, 2668, 2690, 2712, 2734, 2756, 2778, 2800, 2822, 2844, 2866, 2888, 2910, 2932, 2954, 2976, 2998, 3020, 3042, 3064, 3086, 3108, 3130, 3152, 3174, 3196, 3218, 3240, 3262, 3284, 3306, 3331, 3353, 3375, 3397, 3419, 3441, 3463, 3485, 3507, 3529, 3551, 3573, 3595, 3617, 3639, 3661, 3683, 3705, 3727, 3749, 3771, 3793, 3815, 3837, 3859, 3881, 3903, 3925, 3947, 3969, 3991, 4013, 4035, 4057, 4079, 4101, 4123, 4145, 4167, 4189, 4211, 4233, 4255, 4277, 4299, 4321, 4343, 4365, 4387, 4409, 4434, 4456, 4478, 4500, 4522, 4544, 4566, 4588, 4610, 4632, 4654, 4676, 4698, 4720, 4742, 4764, 4786, 4808, 4830, 4852, 4874, 4896, 4918, 4940, 4962, 4984, 5006, 5028, 5050, 5072, 5094, 5116, 5138, 5160, 5182, 5204, 5226, 5248, 5270, 5292, 5314, 5336, 5358, 5380, 5402, 5424, 5446, 5468, 5490, 5512, 5537, 5559, 5581, 5603, 5625, 5647, 5669, 5691, 5713, 5735, 5757, 5779, 5801, 5823, 5845, 5867, 5889, 5911, 5933, 5955, 5977, 5999, 6021, 6043, 6065, 6087, 6109, 6131, 6153, 6175, 6197, 6219, 6241, 6263, 6285, 6307, 6329, 6351, 6373, 6395, 6417, 6439, 6461, 6483, 6505, 6527, 6549, 6571, 6593, 6615, 6640, 6662, 6684, 6706, 6728, 6750, 6772, 6794, 6816, 6838, 6860, 6882, 6904, 6926, 6948, 6970, 6992, 7014, 7036, 7058, 7080, 7102, 7124, 7146, 7168, 7190, 7212, 7234, 7256, 7278, 7300, 7322, 7344, 7366, 7388, 7410, 7432, 7454, 7476, 7498, 7520, 7542, 7564, 7586, 7608, 7630, 7652, 7674, 7696, 7718, 7743, 7765, 7787, 7809, 7831, 7853, 7875, 7897, 7919, 7941, 7963, 7985, 8007, 8029, 8051, 8073, 8095, 8117, 8139, 8161, 8183, 8205, 8227, 8249, 8271, 8293, 8315, 8337, 8359, 8381, 8403, 8425, 8447, 8469, 8491, 8513, 8535, 8557, 8579, 8601, 8623, 8645, 8667, 8689, 8711, 8733, 8755, 8777, 8799, 8821, 8846, 8868, 8890, 8912, 8934, 8956, 8978, 9000, 9022, 9044, 9066, 9088, 9110, 9132, 9154, 9176, 9198, 9220, 9242, 9264, 9286, 9308, 9330, 9352, 9374, 9396, 9418, 9440, 9462, 9484, 9506, 9528, 9550, 9572, 9594, 9616, 9638, 9660, 9682, 9704, 9726, 9748, 9770, 9792, 9814, 9836, 9858, 9880, 9902, 9924, 9949, 9971, 9993, 10015, 10037, 10059, 10081, 10103, 10125, 10147, 10169, 10191, 10213, 10235, 10257, 10279, 10301, 10323, 10345, 10367, 10389, 10411, 10433, 10455, 10477, 10499, 10521, 10543, 10565, 10587, 10609, 10631, 10653, 10675, 10697, 10719, 10741, 10763, 10785, 10807, 10829, 10851, 10873, 10895, 10917, 10939, 10961, 10983, 11005, 11027])

f = open('HB_300_dynamics.out', 'rb')
HB_array = pickle.load(f)

CORE_COUNT = 21
STARTS_PER_CORE = 4

frames = np.arange(0,20000,1) ###CHANGE
split_frames = split(frames,CORE_COUNT)
xyz_outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(read_positions)(split_frames[i]) for i in range(CORE_COUNT))

xyz_list = [item for sublist in xyz_outputs for item in sublist]
xyz_list_og = np.stack(xyz_list)

orientation_categories = np.linspace(0,1,26)
states = ['wet','dry','nonbridge']
WINDOW_SIZE = 5000
INTERMITTENCY = 5
def PW_HB_analysis(start_frames):
    #water donor, polymer acceptor
    psHB_array_og = HB_array[(HB_array[:,1]>11029) & (HB_array[:,3]<11030)]
    #water acceptor, but switch the columns 1 and 3
    water_acceptor = HB_array[(HB_array[:,1]<11030) & (HB_array[:,3]>11029)]
    water_acceptor[:, [1, 3]] = water_acceptor[:, [3, 1]]
    
    psHB_array_og = np.vstack([psHB_array_og,water_acceptor])

    #Solvent-Solvent HBs
    ssHB_array_og = HB_array[(HB_array[:,1]>11029) & (HB_array[:,3]>11029)]

    split_HB_orientations = {'wet':[],'dry':[],'nonbridge':[]}
    split_HB_lifetimes = {'wet':[],'dry':[],'nonbridge':[]}
    
    orientation_acf = [np.zeros(WINDOW_SIZE) for i in orientation_categories[1:]]
    orientation_freqs = [0 for i in orientation_categories[1:]]
    
    orientation_states = {}
    for state in states:
        for category_index in range(len(orientation_categories[1:])):
            orientation_states[state,category_index] = 0

    population_acf = {}
    population_freqs = {}
    for state in states:
        population_acf[state] = np.zeros(WINDOW_SIZE)
        population_freqs[state] = 0

    for start_frame in start_frames:
        first_frame_HBs = {'wet':[],'dry':[],'nonbridge':[]}
        break_frames = {'wet':[],'dry':[],'nonbridge':[]}
        
        psHB_array = psHB_array_og[(psHB_array_og[:,0]>=start_frame) & (psHB_array_og[:,0]<start_frame+WINDOW_SIZE)]
        ssHB_array = ssHB_array_og[(ssHB_array_og[:,0]>=start_frame) & (ssHB_array_og[:,0]<start_frame+WINDOW_SIZE)]

        #only acceptor and donor, where first column is donor (solvent), second is acceptor (polymer)
        HB_start_frame = psHB_array[psHB_array[:,0]==start_frame][:,[1,3]]
        ssHB_start_frame = ssHB_array[ssHB_array[:,0]==start_frame][:,[1,3]]
        for row in HB_start_frame:
            solvent_index = row[0]
            HBlist_p = HB_start_frame[HB_start_frame[:,0]==solvent_index][:,1]
            if max(HBlist_p) - min(HBlist_p) > 25:
                if solvent_index not in ssHB_start_frame:
                    first_frame_HBs['dry'].append(list(row))
                else:
                    first_frame_HBs['wet'].append(list(row))
            else:
                first_frame_HBs['nonbridge'].append(list(row))
        
        remaining_frames = frames[start_frame:start_frame+WINDOW_SIZE]

        HB_frame = psHB_array[:,[0,1,3]]
        all_masks = {}
        for state in states:
            all_masks[state] = [] 
        for key,value in first_frame_HBs.items():
            for HBix,HB in enumerate(value):
                whereequal = (HB_frame[:,[1,2]]==HB)
                HBpresence_frames = HB_frame[np.all(whereequal,axis=1)][:,0]
                
                mask = np.isin(remaining_frames, HBpresence_frames)
                presences_binary = mask.astype(int)
                consecutive_zeros = np.convolve(presences_binary, np.ones(INTERMITTENCY), 'valid') == 0

                if np.any(consecutive_zeros) == False:
                    mask = np.ones(WINDOW_SIZE)
                    break_frame = WINDOW_SIZE + start_frame
                else:
                    break_frame = remaining_frames[np.where(consecutive_zeros == True)][0]
                    mask[break_frame - start_frame:] = 0

                first_consecutive_zeros = np.convolve(presences_binary, np.ones(1), 'valid') == 0
                if np.any(first_consecutive_zeros) == False:
                    first_break_frame = WINDOW_SIZE + start_frame
                else:
                    first_break_frame = remaining_frames[np.where(first_consecutive_zeros == True)][0]

                break_frames[key].append(first_break_frame-start_frame)

                population_acf[key] += mask
                population_freqs[key] += 1

                all_masks[key].append(mask)

        first_frame_HB_orientations = {}

        for state in states:
            first_frame_HB_orientations[state] = [[] for i in first_frame_HBs[state]]        

        xyz_list = xyz_list_og[start_frame:start_frame+WINDOW_SIZE]
        for remaining_index,frame in enumerate(remaining_frames):
            frame_xyz = xyz_list[remaining_index]
            for key,value in first_frame_HBs.items():
                for HBix,HB in enumerate(value):
                    if remaining_index >= break_frames[key][HBix]: #remaining_index >= 1:
                        continue
                    solvent_index = int(HB[0])
                    polymer_atom_index = int(HB[1])
                    solvent_residue = topology.atom(solvent_index).residue.index
                    solvent_atom_ix = np.array([i.index for i in topology.residue(solvent_residue).atoms])
                    solvent_director = (frame_xyz[solvent_atom_ix[1]]-frame_xyz[solvent_atom_ix[0]]) + (frame_xyz[solvent_atom_ix[2]]-frame_xyz[solvent_atom_ix[0]])
                    connect_O1, connect_O2 = connection_oxygens[np.argsort(np.abs(connection_oxygens-polymer_atom_index))[0]],connection_oxygens[np.argsort(np.abs(connection_oxygens-polymer_atom_index))[1]]
                    solvent_polymer_director = frame_xyz[connect_O1] - frame_xyz[connect_O2]
                    cos = np.abs(np.dot(solvent_director,solvent_polymer_director)/norm(solvent_director)/norm(solvent_polymer_director))
                    first_frame_HB_orientations[key][HBix].append(cos)

        # for state in states:
        #     for i in first_frame_HB_orientations[state]:
        #         if i[0] > 0.95:
        #             print(np.mean(i))

        for state,value in first_frame_HB_orientations.items():
            first_frame_HB_orientations[state] = [np.mean(i) if len(i)!=0 else 9999 for i in value]
            for HBix,HBori in enumerate(first_frame_HB_orientations[state]):
                if HBori == 9999:
                    continue
                category_index = [orientation_categories[i]<HBori and orientation_categories[i+1]>=HBori for i in range(0, len(orientation_categories)-1)].index(True)
                orientation_states[state,category_index] += 1
                orientation_acf[category_index] += all_masks[state][HBix]
                orientation_freqs[category_index] += 1
        f = open('{}_ended.eserout'.format(start_frame),'w')
        f.close()
    #orientation_acf: list of 50D arrays
    #orientation_freqs: list of integers
    #population_acf: dictionary of 50D arrays
    #population_freqs: dictionary of integers
    #orientation_states: dictionary of integers
    return (population_acf,population_freqs,orientation_acf,orientation_freqs,orientation_states)

start_frames = [int(i) for i in np.linspace(0,20000-WINDOW_SIZE,STARTS_PER_CORE*CORE_COUNT)]
split_frames = split(start_frames,CORE_COUNT)

outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(PW_HB_analysis)(split_frames[i]) for i in range(CORE_COUNT))

all_population_acf = {}
all_population_freqs = {}
for state in states:
    all_population_acf[state] = np.zeros(WINDOW_SIZE)
    all_population_freqs[state] = 0

all_orientation_acf = [np.zeros(WINDOW_SIZE) for i in orientation_categories[1:]]
all_orientation_freqs = [0 for i in orientation_categories[1:]]

all_orientation_states = {}
for state in states:
    for category_index in range(len(orientation_categories[1:])):
        all_orientation_states[state,category_index] = 0

for output_tuple in outputs:
    output_population_acf = output_tuple[0]
    output_population_freqs = output_tuple[1]
    output_orientation_acf = output_tuple[2]
    output_orientation_freqs = output_tuple[3]
    output_orientation_states = output_tuple[4]

    for state in states:
        all_population_acf[state] += output_population_acf[state]
        all_population_freqs[state] += output_population_freqs[state]
    
    for category_index in range(len(orientation_categories[1:])):
        all_orientation_acf[category_index] += output_orientation_acf[category_index]
        all_orientation_freqs[category_index] += output_orientation_freqs[category_index]

    for state in states:
        for category_index in range(len(orientation_categories[1:])):
            all_orientation_states[state,category_index] += output_orientation_states[state,category_index]

for state in states:
    all_population_acf[state] = list(all_population_acf[state] / all_population_freqs[state])

for category_index in range(len(orientation_categories[1:])):
    all_orientation_acf[category_index] = list(all_orientation_acf[category_index] / all_orientation_freqs[category_index])

f_out = open('orientationvslifetime_v4.out','w')
for state in states:
    f_out.write('acf["{}"] = {}\n'.format(state,all_population_acf[state]))
    f_out.write('population["{}"] = {}\n'.format(state,all_population_freqs[state]))

for category_index in range(len(orientation_categories[1:])):
    f_out.write('orientation_acf[{}] = {}\n'.format(category_index,all_orientation_acf[category_index]))
    f_out.write('orientation_population[{}] = {}\n'.format(category_index,all_orientation_freqs[category_index]))

for state in states:
    for category_index in range(len(orientation_categories[1:])):
        f_out.write('orientation_states["{}",{}] = {}\n'.format(state,category_index,all_orientation_states[state,category_index]))

print(time.time()-sttime)