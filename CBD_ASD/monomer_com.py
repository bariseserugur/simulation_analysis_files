import os
import numpy as np
import time
from joblib import Parallel,delayed
import itertools
import sys
from itertools import combinations
import mdtraj as md
import MDAnalysis as mda
import copy
start = time.time()

topology = md.load('sys.pdb').topology
total_atom_count = topology.n_atoms

POLY_COUNT = 0
if topology.atom(total_atom_count-1).residue.n_atoms != 53:
    POLY_COUNT = len([i for i in topology.residues])
else:
    CBD_residue_name = topology.atom(total_atom_count-1).residue.name
    POLY_COUNT = len([i for i in topology.residues if i.name != CBD_residue_name])

map_element = {'C':12,'N':14,'H':1,'O':16}
map_masses = {'C':12.011,'H':1.008,'O':15.999,'N':14.007}
PVP_pattern = np.array([12,14,12,12,1,1,12,1,1,12,1,1,16,12,1,1,1]) #1 and 14 are backbone, 2,3,4,7,10,13 are side chain
VA_pattern = np.array([12,16,12,16,12,1,1,1,12,1,1,1]) #1,9 are backbone, 2,3,4,5 are side chain
PMMA_pattern = np.array([12,12,1,1,12,12,16,16,12,1,1,1,1,1,1]) #1,2 are backbone, 6,7,8,9 are side chain

HPMCAS2 = np.array([12, 12, 16, 12, 12, 12, 1, 12, 1, 16, 1, 1, 16, 1, 16, 1, 16, 1, 12, 1, 1, 1, 12, 1, 1, 1, 12, 12, 1, 1, 12, 1, 1, 1, 16, 1, 1, 1])[:-1] #HP, 1,2,3,4,5,6 are backbone, 8,10,19 and 13,27,28,31,35 are side
HPMCAS1 = np.array([12, 12, 16, 12, 12, 12, 1, 12, 1, 16, 1, 1, 16, 1, 16, 1, 16, 1, 12, 1, 1, 1, 12, 1, 1, 1, 12, 12, 12, 1, 1, 12, 1, 1, 16, 16, 1, 16, 1])[:-1] #Su, 1,2,3,4,5,6 bb, 8,10,19 are side, 13,27,28,35,29,32,36,38
HPMCAS3 = np.array([12, 12, 16, 12, 12, 12, 1, 12, 1, 16, 1, 1, 16, 1, 16, 1, 16, 1, 12, 1, 1, 1, 12, 1, 1, 1, 12, 12, 1, 1, 1, 16, 1])[:-1] #3Ac, 1,2,3,4,5,6 are backbone, 
                                                                                                                                            #8,10,19 are side, 13,27,28,32 are side
HPMCAS4 = np.array([12, 12, 16, 12, 12, 12, 1, 12, 1, 16, 1, 1, 16, 1, 16, 1, 16, 1, 12, 1, 1, 1, 12, 1, 1, 1, 1, 1])[:-1] #3H, 1,2,3,4,5,6 are backbone, 5,10,19 and 5,15,23 are side
HPMCP3 = np.array([12, 12, 16, 12, 12, 12, 1, 12, 1, 16, 1, 1, 16, 1, 16, 1, 16, 1, 12, 1, 1, 1, 12, 1, 1, 1, 1, 1])[:-1] #3H, 1,2,3,4,5,6 are backbone, 5,10,19 and 5,15,23 are side
HPMCP2 = np.array([12, 12, 16, 12, 12, 12, 1, 12, 1, 16, 1, 1, 16, 1, 16, 1, 16, 1, 12, 1, 1, 1, 12, 1, 1, 1, 12, 12, 1, 1, 12, 1, 1, 1, 16, 1, 1, 1])[:-1]
#HP, 1,2,3,4,5,6 bb, 13,27,28,35,31 side, 13,27,28,31,35 side, 15,23 side
HPMCP1 = np.array([12, 12, 16, 12, 12, 12, 1, 12, 1, 16, 1, 1, 16, 1, 16, 1, 16, 1, 12, 1, 1, 1, 12, 1, 1, 1, 12, 12, 12, 12, 1, 12, 12, 12, 1, 1, 12, 16, 16, 16, 1, 1, 1])[:-1]
#P, 1,2,3,4,5,6 bb, 8,10,19 side, 28,29,30,32,33,34 side


POLY = sys.argv[1]
POLY_patterns = {}
POLY_patterns['PVP'] = [PVP_pattern]
POLY_patterns['PVPVA'] = [PVP_pattern,VA_pattern]
POLY_patterns['PMMA'] = [PMMA_pattern]
POLY_patterns['HPMCAS'] = [HPMCAS1,HPMCAS2,HPMCAS3,HPMCAS4]
POLY_patterns['HPMCP'] = [HPMCP1,HPMCP2,HPMCP3]

POLY_backbones = {}
POLY_backbones['PVP',0] = [[1,14]]
POLY_backbones['PVPVA',0] = [[1,14]]
POLY_backbones['PVPVA',1] = [[1,9]]
POLY_backbones['PMMA',0] = [[1,2]]
POLY_backbones['HPMCAS',0] = [[1,2,3,4,5,6]]
POLY_backbones['HPMCAS',1] = [[1,2,3,4,5,6]]
POLY_backbones['HPMCAS',2]  = [[1,2,3,4,5,6]]
POLY_backbones['HPMCAS',3]  = [[1,2,3,4,5,6]]
POLY_backbones['HPMCP',0]  = [[1,2,3,4,5,6]]
POLY_backbones['HPMCP',1]  = [[1,2,3,4,5,6]]
POLY_backbones['HPMCP',2]  = [[1,2,3,4,5,6]]

POLY_sidechains = {}
POLY_sidechains['PVP',0] = [[2,3,4,7,10,13]]
POLY_sidechains['PVPVA',0] = [[2,3,4,7,10,13]]
POLY_sidechains['PVPVA',1] = [[2,3,4,5]]
POLY_sidechains['PMMA',0] = [[6,7,8,9]]

POLY_sidechains['HPMCAS',0] = [[8,10,19],[13,27,28,35,29,32,36,38]]
POLY_sidechains['HPMCAS',1] = [[8,10,19],[13,27,28,31,35]]
POLY_sidechains['HPMCAS',2]  = [[8,10,19],[13,27,28,32]]
POLY_sidechains['HPMCAS',3]  = [[5,10,19],[5,15,23]]

POLY_sidechains['HPMCP',0]  = [[8,10,19],[28,29,30,32,33,34]]
POLY_sidechains['HPMCP',1]  = [[13,27,28,35,31],[13,27,28,31,35],[15,23]]
POLY_sidechains['HPMCP',2]  = [[5,10,19],[5,15,23]]

def find_monomers(sys_atoms_input):
    monomer_list = []
    monomer_iter = 1
    for chain_no, chain in enumerate(sys_atoms_input):
        if chain_no == 0:
            if POLY.startswith('HPMC'):
                monomer_list.append(-10000)
            monomer_list.append(-10000)
        else:
            if POLY.startswith('HPMC'):
                # monomer_list.append(monomer_list[-1]+1)
                monomer_list.append(-10000)
            # monomer_list.append(monomer_list[-1]+1)
            monomer_list.append(-10000)

        while len(chain) > 0:
            for patternix, pattern in enumerate(POLY_patterns[POLY]):
                if np.array_equal(chain[:len(pattern)], pattern):
                    sides = [0 for i in POLY_backbones[POLY,patternix]]
                    for patternelement, i in enumerate(pattern):
                        if any(patternelement+1 in x for x in POLY_backbones[POLY,patternix]) == False:
                            monomer_list.append(-10000)
                            # sides[patternelement] = -10000
                            # monomer_iter += 1
                        else:
                            whichside = [index for index,obj in enumerate(POLY_backbones[POLY,patternix]) if patternelement+1 in obj][0]
                            
                            if sides[whichside] == 0:
                                sides[whichside] = monomer_iter
                                monomer_list.append(monomer_iter)
                                monomer_iter += 1
                            else:
                                monomer_list.append(sides[whichside])
                    chain = chain[len(pattern):]
                    break
        monomer_iter += 1000
        monomer_list.append(-10000) #monomer_list[-1])
    return monomer_list

f = open('sys.pdb','r').readlines()

sys_atoms = [[] for i in range(POLY_COUNT)]
masses = []
for line in f:
    if 'ATOM' in line:
        molno = int(line.split()[4])
        if molno > POLY_COUNT:
            break
        masses.append(map_masses[line.split()[-1]])
        sys_atoms[molno-1].append(map_element[line.split()[-1]])

if POLY.startswith('HPMC'):
    sys_atoms = [np.array(i[2:-1]) for i in sys_atoms]
else:
    sys_atoms = [np.array(i[1:-1]) for i in sys_atoms]

monomer_list = find_monomers(sys_atoms)


def read_write_header(text_file):
    for i in range(9):
        line = text_file.readline()
        if i == 3:
            line = '{}\n'.format(monomer_count)
        elif 'ITEM: ATOMS' in line:
            line = 'ITEM: ATOMS id type x y z\n'
        output.write(line)
        if i==5:
            Lsplit = line.split()
            L = float(Lsplit[1]) - float(Lsplit[0])
    return L

output = open('back_com.lammpstrj','w')
analysis_file = 'modified_310.lammpstrj'
N = len(masses)
frame_count = int(int(os.popen('wc -l {}'.format(analysis_file).read().split()[0])/(total_atom_count+9))

selected_indices = [index for index, obj in enumerate(monomer_list) if obj != -10000]
masses = [masses[index] for index, obj in enumerate(monomer_list) if obj != -10000]
masses = np.array([masses,]*3).transpose()
monomer_count = len(list(set([i for i in monomer_list if i != -10000])))
unique_monomers = [i for i in sorted(list(set(monomer_list))) if i != -10000]
monomer_molecules = [monomer_list.count(i) for i in unique_monomers]

with open(analysis_file,'r') as text_file:
    for i in range(frame_count):
        L = read_write_header(text_file)
        halfL = L/2
        frame_coords = np.zeros((N,3))
        selected_count = 0
        for atom_iter in range(total_atom_count):
            line = text_file.readline()
            if atom_iter in selected_indices:
                frame_coords[selected_count] = np.array([float(x) for x in line.split()[-3:]])
                selected_count += 1
        
        molecule_index = 1
        atoms_done = 0
        for j in unique_monomers:
            atomspermolecule = monomer_list.count(j)
            molecule_coords = np.zeros((atomspermolecule,3))
            for k in range(atomspermolecule):
                molecule_coords[k] = frame_coords[atoms_done+k]
            atoms_done += atomspermolecule
            first_atom = copy.deepcopy(molecule_coords[0])
            for k in range(3):
                molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])>(halfL),molecule_coords[:,k]-L,molecule_coords[:,k])
                molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])<(-halfL),molecule_coords[:,k]+L,molecule_coords[:,k])
            molecule_coords = molecule_coords * masses[atoms_done-atomspermolecule:atoms_done]
            molecule_com = np.sum(molecule_coords,axis=0)/np.sum(masses[atoms_done-atomspermolecule:atoms_done][:,0])
            output.write('{} 1 {} {} {}\n'.format(molecule_index,molecule_com[0],molecule_com[1],molecule_com[2]))
            molecule_index += 1
    output.close()

modif_init = open('modified_sys.pdb','r')

com_init = open('back_com_init.pdb','w')
for i in range(3):
    line = modif_init.readline()
    com_init.write(line)

for index,obj in enumerate(unique_monomers):
    line = 'ATOM{:>7} {:<5}AAA{:>6}'.format(index+1,1,obj)+'      12.389  19.053  34.593  1.00  0.00           \n'
    com_init.write(line)
