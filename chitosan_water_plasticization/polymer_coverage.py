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
from scipy.stats import sem
from label_monomers import label_monomers

'''
This code calculates the fraction of monomers that have at least one H-bond. Depending on whether the "coverage_type" variable is monomer or HBsite, it calculates coverage of monomers containing a H-bond, or H-bond sites.
'''

SOLVENT = str(sys.argv[1])
if SOLVENT == 'glycerol':
    raise Exception('It doesnt handle glycerol yet!')


coverage_type = 'HBsite'

allcovgs = []
sems = []
for conc in [0,1,5,10,15,20,30,40]:
    covgs = []
    for simno in [1,2,3,4]:
        print(conc)
        os.chdir('/scratch/gpfs/bu9134/chitosan_water_900_200/{}/{}'.format(conc,simno))
        topology = md.load('sys_evaporated.pdb').topology
        
        if coverage_type == 'HBsite':
            HBsites = len([i for i in range(11030) if topology.atom(i).element.symbol in ['O','N']])
        if coverage_type == 'monomer':
            HBsites = 500
            index_to_monomer = label_monomers()

        HB_f = open('HB_300_dynamics.out','rb')

        #Retrieve the full list of H-bonds
        HB_array = pickle.load(HB_f)

        #only display H-bonds involving a polymer
        HB_array = HB_array[(HB_array[:,1]<11030) | (HB_array[:,3]<11030)]

        frames = np.arange(0,int(max(HB_array[:,0])),100)

        coverages = []
        for frameix, frame in enumerate(frames):
            HB_frame = HB_array[HB_array[:,0]==frame][:,[1,3]] #only acceptor and donor
            HB_frame = HB_frame[HB_frame<11030] #only polymers

            if coverage_type == 'HBsite':
                unique_atoms = np.unique(HB_frame)

            if coverage_type == 'monomer':
                HB_frame = np.vectorize(index_to_monomer.get)(HB_frame)
                unique_atoms = np.unique(HB_frame)
            
            coverage = len(unique_atoms)
            coverages.append(coverage/HBsites)
        covgs.append(np.mean(coverages))
    allcovgs.append(np.mean(covgs))
    sems.append(sem(covgs))
print(allcovgs)
print(sems)
        




#Output File
# f_out = open('HB_orientation.out','w')
# f_out.write('average_orientation = {}\n'.format(average_orientation))
# f_out.write('freqs = {}\n'.format(freqs))
# f_out.write('vals = {}\n'.format(vals))
# f_out.write('meanON = {} #0 is Oxygen, 1 is Nitrogen H-bonds \n'.format(meanON))