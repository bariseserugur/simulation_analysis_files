import os
import numpy as np
import time
import itertools
import sys

start = time.time()

dirname = sys.argv[1]
simno = int(sys.argv[2])


kb = 1.380649e-23
etocoulomb = 1.60217646e-19
Jtohartree = 2.2937104486906e17
P = 32
dipole_output = open('/scratch/gpfs/bu9134/dielectric_outputs/{}_{}.di'.format(dirname,simno),'w')
path = '/scratch/gpfs/bu9134/RPMD_TAFFI_Results'

M2_dict = dict()
dipoles_dict = dict()

def dielectric_calculator(dirname,simno):
    def read_header(text_file):
        for i in range(9):
            line = text_file.readline()
            if i==5:  
                Lsplit = line.split()
                L = float(Lsplit[1]) - float(Lsplit[0])
        return L
        
    charges = []
    atom_classes = []
    type_to_class = dict()
    ff_file = open(path+'/{}/ff_{}.xml'.format(dirname,dirname),'r')
    ff_lines = ff_file.readlines()
    for ff_line in ff_lines:
        if '<Atom name=' in ff_line:
            atom_classes.append(ff_line.split('"')[-2])
        if '<Atom type' in ff_line:
            type_to_class[ff_line.split('"')[1]] = float(ff_line.split('"')[3])
    for i in atom_classes:
        charges.append(type_to_class[i])
    atomspermolecule = len(charges)
    atomcount = int(np.ceil(5000/atomspermolecule)) * atomspermolecule

    if os.path.isfile(path+'/{}/{}/Ptrajectory.lammpstrj'.format(dirname,simno)):
        files = [path+'/{}/{}/Ptrajectory.lammpstrj'.format(dirname,simno)] 
        linecount = float(os.popen('wc -l {}/"{}"/{}/Ptrajectory.lammpstrj'.format(path,dirname,simno)).read().split()[0])
        frames = [1000]
        file_split = 1
    elif os.path.isfile(path+'/{}/{}/Ptrajectory2_1.lammpstrj'.format(dirname,simno)):
        file_split = 2
    elif os.path.isfile(path+'/{}/{}/Ptrajectory4_1.lammpstrj'.format(dirname,simno)):
        file_split = 4
    if file_split > 1:
        files = [path+'/{}/{}/Ptrajectory{}_{}.lammpstrj'.format(dirname,simno,file_split,file_no) for file_no in range(1,file_split+1)]
        linecounts = [float(os.popen('wc -l {}/"{}"/{}/Ptrajectory{}_{}.lammpstrj'.format(path,dirname,simno,file_split,file_no)).read().split()[0]) for file_no in range(1,file_split+1)]
        frames = [line_count/((atomcount*P)+9) for line_count in linecounts]

    charges = np.array([charges,]*3).transpose()
    moleculecount = int(atomcount/atomspermolecule)
    frame_count = 1000
    dipoles = np.zeros((frame_count,3))
    M2list = []
    frame_no = 0
    for index,obj in enumerate(files):
        with open(obj,'r') as text_file:
            for i in range(int(frames[index])):
                L = read_header(text_file)
                halfL = L/2
                N = atomcount*P
                frame_coords = np.zeros((N,3))
                for atom_iter in range(N):
                    line = text_file.readline()
                    frame_coords[atom_iter] = np.array([float(x) for x in line.split()[-3:]])
                dipole_sums = np.zeros(3)
                for bead_no in range(P):
                    totaldipole = np.zeros(3)
                    for j in range(moleculecount):
                        molecule_coords = np.zeros((atomspermolecule,3))
                        for k in range(atomspermolecule):
                            molecule_coords[k] = frame_coords[j*P*atomspermolecule+(P*k+bead_no)]
                        first_atom = molecule_coords[0].copy()
                        for k in range(3):
                            molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])>(halfL),molecule_coords[:,k]-L,molecule_coords[:,k])
                            molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])<(-halfL),molecule_coords[:,k]+L,molecule_coords[:,k])
                        molecule_coords = molecule_coords * charges
                        totaldipole += np.sum(molecule_coords,axis=0)
                    dipole_sums += totaldipole
                frame_dipole = dipole_sums/P
                dipoles[frame_no] = frame_dipole
                M2list.append(np.sum(frame_dipole**2))
                frame_no += 1
    M2_dict[dirname,simno] = M2list
    dipoles_dict[dirname,simno] = dipoles
    return dipoles, M2list

dipoles_dict[dirname,simno],M2_dict[dirname,simno] = dielectric_calculator(dirname,simno)
dipole_output.write('rpmd_dipoles_dict["{}",{}] = {}\n'.format(dirname,simno,[list(i) for i in dipoles_dict[dirname,simno]]))
dipole_output.write('rpmd_M2_dict["{}",{}] = {}\n'.format(dirname,simno,M2_dict[dirname,simno]))

dipole_output.close()

end = time.time()
print(end-start)