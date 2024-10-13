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

'''Divide every CBD molecule into three functional groups (compatibilizer, hydrophilic, tail), and create a .pdb file with center of mass of these groups, 
along with the oxygens in every polymer. The resulting .pdb can be used for radial distribution function analysis.'''

POLY = sys.argv[1]

topology = md.load('sys.pdb').topology
total_atom_count = topology.n_atoms

if topology.atom(total_atom_count-1).residue.n_atoms != 53:
    last_CBD_index = total_atom_count #atom with this index doesn't exist, this deals with the scenarios for 100% polymer systems
else:
    CBD_residue_name = topology.atom(total_atom_count-1).residue.name
    last_CBD_index = min([i.index for i in topology.atoms if i.residue.name == CBD_residue_name])

CBD_groups = {}
#below, first atom is 1, not 0. that's why the -1 correction at the end
CBD_groups['tail'] = np.array([16,39,40,18,44,43,19,45,46,20,47,48,21,50,51,49]) - 1
CBD_groups['hydrophilic'] = np.array([2,11,13,14,15,3,23,53,38,37,22,52]) - 1
CBD_groups['compatibilizer'] = np.array([i for i in np.arange(0,53,1) if i not in CBD_groups['tail'] and i not in CBD_groups['hydrophilic']])

HPMCAS_O = ['11', '20', '30', '23', '21', '5', '12', '27', '1']
HPMCP_O = ['5', '26', '16', '1', '27', '31', '10', '28']

#indices of polymer oxygens, and CBD groups
all_groups = []
group_types = []
for atom in topology.atoms:
    if atom.residue.name != CBD_residue_name and atom.element.symbol == 'O':
        if POLY == 'HPMCAS' and atom.name not in HPMCAS_O:
            continue
        if POLY == 'HPMCP' and atom.name not in HPMCP_O:
            continue
        all_groups.append([atom.index])
        group_types.append('O')
for atom in topology.atoms:
    if atom.residue.name == CBD_residue_name and list(atom.residue.atoms).index(atom) == 0:
        all_groups.append(list(CBD_groups['tail'] + atom.index))
        group_types.append('tail')
        all_groups.append(list(CBD_groups['hydrophilic'] + atom.index))
        group_types.append('hydrophilic')
        all_groups.append(list(CBD_groups['compatibilizer'] + atom.index))
        group_types.append('compatibilizer')

type_to_number = {'O':1,'tail':2,'hydrophilic':3,'compatibilizer':4}

map_masses = {'C':12.011,'H':1.008,'O':15.999,'N':14.007}

f = open('sys.pdb','r').readlines()

masses = []
for line in f:
    if 'ATOM' in line:
        masses.append(map_masses[line.split()[-1]])
        
def read_write_header(text_file):
    for i in range(9):
        line = text_file.readline()
        if i == 3:
            line = '{}\n'.format(len(all_groups))
        elif 'ITEM: ATOMS' in line:
            line = 'ITEM: ATOMS id type x y z\n'
        output.write(line)
        if i==5:
            Lsplit = line.split()
            L = float(Lsplit[1]) - float(Lsplit[0])
    return L

output = open('cbd_com.lammpstrj','w')
analysis_file = '21step.lammpstrj'
N = len(masses)
frame_count = int(int(os.popen('wc -l {}'.format(analysis_file)).read().split()[0])/(total_atom_count+9))

masses = np.array([masses,]*3).transpose()

with open(analysis_file,'r') as text_file:
    for i in range(frame_count):
        L = read_write_header(text_file)
        halfL = L/2
        frame_coords = np.zeros((N,3))
        for atom_iter in range(total_atom_count):
            line = text_file.readline()
            frame_coords[atom_iter] = np.array([float(x) for x in line.split()[-3:]])
        
        molecule_index = 1
        for group in all_groups:
            atomspermolecule = len(group)
            molecule_coords = np.zeros((atomspermolecule,3))
            for index,obj in enumerate(group):
                molecule_coords[index] = frame_coords[obj]
            first_atom = copy.deepcopy(molecule_coords[0])
            for k in range(3):
                molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])>(halfL),molecule_coords[:,k]-L,molecule_coords[:,k])
                molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])<(-halfL),molecule_coords[:,k]+L,molecule_coords[:,k])
            molecule_coords = molecule_coords * masses[group]
            molecule_com = np.sum(molecule_coords,axis=0)/np.sum(masses[group][:,0])
            output.write('{} {} {} {} {}\n'.format(molecule_index,type_to_number[group_types[molecule_index-1]],molecule_com[0],molecule_com[1],molecule_com[2]))
            molecule_index += 1
    output.close()

modif_init = open('sys.pdb','r')

com_init = open('cbd_com.pdb','w')
for i in range(3):
    line = modif_init.readline()
    com_init.write(line)

for index,obj in enumerate(all_groups):
    line = 'ATOM{:>7} {:<5}AAA{:>6}'.format(index+1,type_to_number[group_types[index]],1)+'      12.389  19.053  34.593  1.00  0.00           \n'
    com_init.write(line)