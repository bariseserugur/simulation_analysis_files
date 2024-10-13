from __future__ import print_function
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmm.app.internal.unitcell import computeLengthsAndAngles
import re
import numpy as np
from scipy.stats import sem
from openmmtools import integrators
import time
import sys
import os
import copy
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import joblib
from joblib import Parallel,delayed
import threading
from multiprocessing import Pool
import mdtraj as md

sys.setrecursionlimit(100000)

'''
This script takes in the system, and iterates through every molecule and in each iteration, removes all other molecules and calculates
the vapor state energy. Then, it calculates the difference between the sum of vapor phase energies and the ensemble liquid energy and divides it by 
system volume to obtain the Hildebrand solubility parameter in kJ/cm3.
'''
def minimum_distances_to_polymer(all_coords,water_oxygens,poly_indices,L): 
    distances = []
    poly_coords = all_coords[poly_indices]
    for oxygen_index in water_oxygens:
        coord = all_coords[oxygen_index]
        dists = dist_pbc(poly_coords, coord, L)**0.5
        distances.append(min(dists))
    return np.array(distances)

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


TIMESTEP = 0.0000001 #picoseconds
TEMPERATURE = 300
TIMESTEP = 0.00025 #picoseconds
THERMO_FREQ = 1/TIMESTEP #every 1 ps
BAROSTAT_FRQ = 0.025/TIMESTEP #every 25 fs
#####

platform = Platform.getPlatformByName('CPU')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";
properties = {}

pdb = PDBFile('sys_evaporated.pdb') 
ff_file = 'ff_sys.xml'
forcefield = ForceField(ff_file)

##### Prepare System
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=10*angstrom,rigidWater=False, removeCMMotion=True)
forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
nonbonded_force = forces['NonbondedForce']
nonbonded_force.setUseSwitchingFunction(True)
nonbonded_force.setSwitchingDistance(9*angstrom)

barostat = MonteCarloBarostat(1*bar,TEMPERATURE*kelvin,BAROSTAT_FRQ)
system.addForce(barostat)
integrator = LangevinIntegrator(TEMPERATURE*kelvin,1/picosecond,TIMESTEP*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

Restart.load_simulation('300_equil.save',simulation,'classical')

total_energy = simulation.context.getState(getEnergy=True, enforcePeriodicBox=True).getPotentialEnergy()._value
pos = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
pos_copy = copy.deepcopy(pos)

state_og = copy.deepcopy(simulation.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox=True))
L = state_og.getPeriodicBoxVectors(asNumpy=True)[0][0]._value
pos_og = np.array(state_og.getPositions()._value)
vel_og = np.array(state_og.getVelocities()._value)

water_bond_r0 = 0.1012
water_bond_k = 443153.38080000004
water_angle_theta0 = 1.9764108449583786
water_angle_k = 317.5656

def removal_energy(water_residues_batch):
    removal_energies = []
    for water_residue in water_residues_batch:
        modeller = Modeller(pdb.topology, pos_copy)

        toDeleteix = np.array([x.index for x in modeller.topology.atoms() if x.residue.index == water_residue.index])
        toDelete = [x for x in modeller.topology.atoms() if x.residue.index == water_residue.index]

        deletepos = pos_og[toDeleteix]

        pos = np.delete(pos_og, toDeleteix, axis=0)
        vel = np.delete(vel_og, toDeleteix, axis=0)
        modeller.delete(toDelete)

        system = forcefield.createSystem(topology=modeller.topology, nonbondedMethod=PME,nonbondedCutoff=10*angstrom,rigidWater=False, removeCMMotion=True)
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        nonbonded_force = forces['NonbondedForce']
        nonbonded_force.setUseSwitchingFunction(True)
        nonbonded_force.setSwitchingDistance(9*angstrom)

        # barostat = MonteCarloBarostat(1*bar,TEMPERATURE*kelvin,BAROSTAT_FRQ)
        # system.addForce(barostat)
        platform = Platform.getPlatformByName('CPU')
        platform = Platform.getPlatformByName("Reference")
        properties = {}

        integrator = LangevinIntegrator(TEMPERATURE*kelvin,1/picosecond,TIMESTEP*picoseconds)
        simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

        simulation.context.setPositions(pos)
        simulation.context.setVelocities(vel)
        simulation.context.setPeriodicBoxVectors(np.array([L,0,0]),np.array([0,L,0]),np.array([0,0,L]))
        
        total_energy_minus_water = simulation.context.getState(getEnergy=True, enforcePeriodicBox=True).getPotentialEnergy()._value

        ba = deletepos[1] - deletepos[0]
        bc = deletepos[2] - deletepos[0]

        d1 = numpy.linalg.norm(deletepos[1]-deletepos[0])
        d2 = numpy.linalg.norm(deletepos[2]-deletepos[0])

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        bond_energy = 0
        bond_energy += 0.5 * water_bond_k * (d1-water_bond_r0)**2
        bond_energy += 0.5 * water_bond_k * (d2-water_bond_r0)**2

        angle_energy = 0.5 * water_angle_k * (angle-water_angle_theta0)**2

        removal_energy = (bond_energy + angle_energy) + (total_energy_minus_water) - (total_energy)
        removal_energies.append(removal_energy)
    return np.array(removal_energies)

topology = md.load('sys_evaporated.pdb').topology
water_oxygens = np.array([i.index for i in topology.atoms if i.index > 11029 and i.element.symbol == 'O'])
poly_indices = [i.index for i in topology.atoms if i.index < 11030 and i.element.symbol in ['O','C']]

all_coords = pos_og
minimum_distances = minimum_distances_to_polymer(all_coords,water_oxygens,poly_indices,L)
polymer_waters = water_oxygens[minimum_distances<0.35]

# water_residues = [i.residue for i in pdb.topology.atoms() if i.index > 11029][::3]
water_residues = [i.residue for i in pdb.topology.atoms() if i.index in polymer_waters]

CORE_COUNT = 24
water_residues_cores = split(water_residues,CORE_COUNT)

pool = Pool(processes=CORE_COUNT)

inputs = [(water_residues_cores[i]) for i in range(CORE_COUNT)]
outputs = pool.map(removal_energy, inputs)
removal_energies = np.hstack(outputs)

water_removal_energy = np.mean(removal_energies)

f_out = open('water_removal_energy.out','w')
f_out.write('water_removal_energy = {}\n'.format(water_removal_energy))
f_out.write('distribution = {}'.format(list(removal_energies)))
f_out.close()

# outputs = Parallel(n_jobs=CORE_COUNT, backend='threading')(delayed(removal_energy)(water_residues_cores[i]) for i in range(CORE_COUNT))

# removal_energies = np.hstack(outputs)

