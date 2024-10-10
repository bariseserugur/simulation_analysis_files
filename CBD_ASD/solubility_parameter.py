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

'''
This script takes in the system, and iterates through every molecule and in each iteration, removes all other molecules and calculates
the vapor state energy. Then, it calculates the difference between the sum of vapor phase energies and the ensemble liquid energy and divides it by 
system volume to obtain the Hildebrand solubility parameter in kJ/cm3.
'''

temp_range = []
current_dir = os.getcwd()
for filename in os.listdir(current_dir):
    if 'extended' in filename and 'save' in filename:
        temp_range.append(int(filename[:3]))
temp_range = sorted(temp_range)

TIMESTEP = 0.0000001 #picoseconds
TEMPERATURE = 310
#####

platform = Platform.getPlatformByName('CPU')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";
properties = {}

pdb = PDBFile('sys.pdb') 
ff_file = 'ff_sys.xml'
forcefield = ForceField(ff_file)
modeller = Modeller(pdb.topology, pdb.positions)
modeller_og = copy.deepcopy(modeller)

sys.setrecursionlimit(100000)

box_size = float(open('{}_equil.save'.format(TEMPERATURE),'r').readlines()[1].split()[0])

system = forcefield.createSystem(topology=modeller_og.topology, nonbondedMethod=PME,nonbondedCutoff=(box_size/2)*nanometer,rigidWater=False, removeCMMotion=True)
system = Geometric.GeometricMix(system,0.5)
for i in range(system.getNumForces()):
    force = system.getForce(i)
    force.setForceGroup(i)
forces = {system.getForce(index).__class__.__name__: system.getForce(
            index) for index in range(system.getNumForces())}
nonbonded_force = forces['NonbondedForce']
geometric_nonbonded_force = forces['CustomNonbondedForce']
geometric_nonbonded_force.setUseLongRangeCorrection(False)
nonbonded_force.setUseDispersionCorrection(False)
nonbonded_force.setUseSwitchingFunction(False)
# nonbonded_force.setExceptionsUsePeriodicBoundaryConditions(True)

no_epsilon = False
no_charge = False
if no_charge == True:
    #no charge
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        nonbonded_force.setParticleParameters(index, charge*0, sigma, epsilon)
    for i in range(nonbonded_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
        nonbonded_force.setExceptionParameters(i, p1, p2, q*0, sig, eps)

if no_epsilon == True:
    # no epsilon
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)

        sigma, epsilon = geometric_nonbonded_force.getParticleParameters(index)
        geometric_nonbonded_force.setParticleParameters(index, [sigma, epsilon * 0])

    for i in range(nonbonded_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
        if eps._value != 0.0:
            nonbonded_force.setExceptionParameters(i, p1, p2, q, sig, eps*0)

integrator = LangevinIntegrator(TEMPERATURE*kelvin,1/picosecond,TIMESTEP*picoseconds)
simulation = Simulation(topology=modeller_og.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
Restart.load_simulation('{}_equil.save'.format(TEMPERATURE),simulation,'classical')

liquid_energy = simulation.context.getState(getEnergy=True,groups={3,5}).getPotentialEnergy()._value #in kJ/mol
print(liquid_energy)
state_og = copy.deepcopy(simulation.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox=True))
vectors = np.array(state_og.getPeriodicBoxVectors(asNumpy=True))
system_volume = (vectors[0][0]**3) * (10**-21) * (6.02214076 * (10**23)) #cm3/mol
vectors[0][0], vectors[1][1], vectors[2][2] = 1000,1000,1000

gas_energies = 0
for residue in modeller.topology.residues():
    modeller = Modeller(pdb.topology, pdb.positions)
    init_time = time.time()
    
    toDeleteix = [x.index for x in modeller.topology.atoms() if x.residue.index!=residue.index]
    toDelete = [x for x in modeller.topology.atoms() if x.residue.index!=residue.index]

    pos = np.array(state_og.getPositions()._value)
    pos = np.delete(pos, toDeleteix, axis=0) + 100 #place it in the middle of the box by adding the +100 term
    vel = np.array(state_og.getVelocities()._value)
    vel = np.delete(vel, toDeleteix, axis=0)
    modeller.delete(toDelete)
    
    modeller.topology.setPeriodicBoxVectors(vectors)

    system = forcefield.createSystem(topology=modeller.topology, nonbondedMethod=CutoffPeriodic,nonbondedCutoff=400*nanometer,rigidWater=False, removeCMMotion=True)
    system = Geometric.GeometricMix(system,0.5)
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
    forces = {system.getForce(index).__class__.__name__: system.getForce(
                index) for index in range(system.getNumForces())}

    nonbonded_force = forces['NonbondedForce']
    geometric_nonbonded_force = forces['CustomNonbondedForce']
    geometric_nonbonded_force.setUseLongRangeCorrection(False)
    nonbonded_force.setUseDispersionCorrection(False)
    nonbonded_force.setUseSwitchingFunction(False)
    # nonbonded_force.setExceptionsUsePeriodicBoundaryConditions(True)
    if no_charge == True:
        #no charge
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            nonbonded_force.setParticleParameters(index, charge*0, sigma, epsilon)
        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            nonbonded_force.setExceptionParameters(i, p1, p2, q*0, sig, eps)

    if no_epsilon == True:
        # no epsilon
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)

            sigma, epsilon = geometric_nonbonded_force.getParticleParameters(index)
            geometric_nonbonded_force.setParticleParameters(index, [sigma, epsilon * 0])

        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            if eps._value != 0.0:
                nonbonded_force.setExceptionParameters(i, p1, p2, q, sig, eps*0)
    
    integrator = LangevinIntegrator(TEMPERATURE*kelvin,1/picosecond,TIMESTEP*picoseconds)
    simulation = Simulation(topology=modeller.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
    simulation.context.setPositions(pos)
    simulation.context.setVelocities(vel)
    simulation.context.setPeriodicBoxVectors(vectors[0],vectors[1],vectors[2])

    residue_energy = simulation.context.getState(getEnergy=True,groups={3,5}).getPotentialEnergy()._value #in kJ/mol
    gas_energies += residue_energy
print(gas_energies)
deltaE = (gas_energies - liquid_energy) #kJ/mol


if deltaE < 0:
    multiplier = -1
else:
    multiplier = 1

hildebrand = multiplier* np.sqrt(np.abs(deltaE)/system_volume * 1000) #1kJ/cm3 = 1000 MPa, so hildebrand variable is in MPa^0.5

f_out = open('hildebrand.out','w')
f_out.write('hildebrand = {}\n'.format(hildebrand))