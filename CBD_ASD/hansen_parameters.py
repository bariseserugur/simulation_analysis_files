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
sys.setrecursionlimit(100000)

'''
Read the input system and iterate through every molecule, and in each iteration, remove all other molecules and calculate
the vapor state energy. Then, calculate the difference between the sum of vapor phase energies and the ensemble liquid energy and divide it by 
system volume to obtain the Hildebrand solubility parameter in kJ/cm3.
'''

TIMESTEP = 0.0000001 #picoseconds
TEMPERATURE = 310
f_out = open('hildebrand.out','w')
#####

platform = Platform.getPlatformByName('CPU')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";
properties = {}

analysis_file = 'modified_310.lammpstrj'

u = mda.Universe('sys.pdb',analysis_file, format='LAMMPSDUMP', dt=0.01)

def HB_analysis(CUTOFF_DISTANCE = 3.5,CUTOFF_ANGLE = 150):
    HB = HydrogenBondAnalysis(
    universe=u,
    donors_sel='element O or element N',
    hydrogens_sel='element H',
    acceptors_sel='element O or element N',
    d_a_cutoff=CUTOFF_DISTANCE,
    d_h_a_angle_cutoff=CUTOFF_ANGLE,
    update_selections=False)

    HB.run(
    start=None,
    stop=None,
    step=None,
    verbose=True)

    HB_results = HB.results.hbonds
    return HB_results

HB_array = HB_analysis()

hansen_parameters = ['combined','HB','LJ','Coulomb']

hansen_dict = {}
for hansen_parameter in hansen_parameters:
    hansen_dict[hansen_parameter] = []

for frame_no in range(0,10):#[299,599,899,1199,1499,1799,2099,2399,2699,2999]:
    frame = u.trajectory[frame_no]

    HB_pairs = []
    for i in HB_array[HB_array[:,0]==frame_no]:
        HB_pairs.append(sorted([i[1],i[3]]))
        HB_pairs.append(sorted([i[2],i[3]]))

    for hansen_parameter in hansen_parameters:
        if hansen_parameter == 'LJ':
            no_HB = True
            no_charge = True
            no_epsilon = False
        elif hansen_parameter == 'Coulomb':
            no_HB = True
            no_epsilon = True
            no_charge= False
        elif hansen_parameter == 'HB':
            no_charge = True
            no_epsilon = True
            no_HB = False
        elif hansen_parameter == 'combined':
            no_charge = False
            no_epsilon = False
            no_HB = False

        # load_file = open('{}_equil.save'.format(TEMPERATURE),'r').readlines()
        # box_size = float(load_file[1].split()[0])
        box_size = frame.dimensions[0]/10

        pdb = PDBFile('sys.pdb') 
        ff_file = 'ff_sys.xml'
        forcefield = ForceField(ff_file)

        topo = pdb.topology
        topo.setPeriodicBoxVectors(np.array([[box_size,0,0],[0,box_size,0],[0,0,box_size]]))

        modeller = Modeller(topo, pdb.positions)
        modeller_copy = copy.deepcopy(modeller)

        system = forcefield.createSystem(topology=modeller_copy.topology, nonbondedMethod=CutoffPeriodic,nonbondedCutoff=((box_size/2))*nanometer,rigidWater=False, removeCMMotion=True)
        system = Coulomb.No_Long(system)
        system = Geometric.GeometricMix(system,0.5)
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            force.setForceGroup(i)

        forces = {system.getForce(index).__class__.__name__: system.getForce(
                    index) for index in range(system.getNumForces())}
        
        nonbonded_force = forces['NonbondedForce']
        geometric_nonbonded_force = system.getForce(6)
        coulomb_force = system.getForce(5)

        geometric_nonbonded_force.setUseLongRangeCorrection(False)
        coulomb_force.setUseLongRangeCorrection(False)
        nonbonded_force.setUseDispersionCorrection(False)
        nonbonded_force.setUseSwitchingFunction(False)
        nonbonded_force.setExceptionsUsePeriodicBoundaryConditions(True)

        nonbonded_copy = copy.deepcopy(nonbonded_force)
        geometric_copy = copy.deepcopy(geometric_nonbonded_force)
        coulomb_copy = copy.deepcopy(coulomb_force)

        if no_charge == True:
            for index in range(nonbonded_force.getNumParticles()):
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
                nonbonded_force.setParticleParameters(index, charge*0, sigma, epsilon)
                charge = coulomb_force.getParticleParameters(index)
                coulomb_force.setParticleParameters(index, (0,))
            for i in range(nonbonded_force.getNumExceptions()):
                (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
                nonbonded_force.setExceptionParameters(i, p1, p2, q*0, sig, eps)

        if no_epsilon == True:
            for index in range(nonbonded_force.getNumParticles()):
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
                nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)
                sigma, epsilon = geometric_nonbonded_force.getParticleParameters(index)
                geometric_nonbonded_force.setParticleParameters(index, [sigma, epsilon * 0])

            for i in range(nonbonded_force.getNumExceptions()):
                (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
                nonbonded_force.setExceptionParameters(i, p1, p2, q, sig, eps*0)

        if no_HB == True:
            for pair in HB_pairs:
                nonbonded_force.addException(pair[0], pair[1], 0, 1, 0)
                geometric_nonbonded_force.addExclusion(pair[0],pair[1])
                coulomb_force.addExclusion(pair[0],pair[1])

        if hansen_parameter == 'HB':
            for pair in HB_pairs:
                sig1,eps1 = geometric_copy.getParticleParameters(int(pair[0]))
                sig2,eps2 = geometric_copy.getParticleParameters(int(pair[1]))
                charge1 = coulomb_copy.getParticleParameters(int(pair[0]))[0]
                charge2 = coulomb_copy.getParticleParameters(int(pair[1]))[0]
                sig_mix = np.sqrt(sig1*sig2)
                eps_mix = np.sqrt(eps1*eps2)
                charge_mix = charge1*charge2
                nonbonded_force.addException(pair[0], pair[1], charge_mix, sig_mix, eps_mix)
                geometric_nonbonded_force.addExclusion(pair[0],pair[1])
                coulomb_force.addExclusion(pair[0],pair[1])

        integrator = LangevinIntegrator(TEMPERATURE*kelvin,1/picosecond,TIMESTEP*picoseconds)
        simulation = Simulation(topology=modeller_copy.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
        # Restart.load_simulation('{}_equil.save'.format(TEMPERATURE),simulation,'classical')

        # pos = simulation.context.getState(getPositions=True).getPositions()
        # pos_copy = copy.deepcopy(pos)
        pos = frame.positions/10
        pos_copy = copy.deepcopy(pos)
        simulation.context.setPositions(pos)
        simulation.context.setPeriodicBoxVectors(np.array([box_size,0,0]),np.array([0,box_size,0]),np.array([0,0,box_size]))

        liquid_energy = simulation.context.getState(getEnergy=True,groups={3,5,6}).getPotentialEnergy()._value #in kJ/mol
        
        state_og = copy.deepcopy(simulation.context.getState(getPositions = True, getVelocities = True, enforcePeriodicBox=True))
        system_volume = (box_size**3) * (10**-21) * (6.02214076 * (10**23)) #cm3/mol

        gas_energies = 0
        for residue in list(modeller.topology.residues()):
            modeller = Modeller(topo, pos_copy)
            
            toDeleteix = np.array([x.index for x in modeller.topology.atoms() if x.residue.index!=residue.index])
            toDelete = [x for x in modeller.topology.atoms() if x.residue.index!=residue.index]
            noDeleteix = np.array([x.index for x in modeller.topology.atoms() if x.index not in toDeleteix])
            minnoDelete = min(noDeleteix)

            postdeleteHBpairs = []
            for pair in HB_pairs:
                empty = []
                a = 0
                for j in pair:
                    if j in toDeleteix:
                        a=1
                        break
                    empty.append(j - minnoDelete)
                if a==0:
                    postdeleteHBpairs.append(empty)

            pos = np.array(state_og.getPositions()._value)
            pos = np.delete(pos, toDeleteix, axis=0) + 450
            vel = np.array(state_og.getVelocities()._value)
            vel = np.delete(vel, toDeleteix, axis=0)
            modeller.delete(toDelete)

            gas_box_size = 1000
            modeller.topology.setPeriodicBoxVectors([np.array([gas_box_size,0,0]),np.array([0,gas_box_size,0]),np.array([0,0,gas_box_size])])

            system = forcefield.createSystem(topology=modeller.topology, nonbondedMethod=NoCutoff,nonbondedCutoff=(gas_box_size/2)*nanometer,rigidWater=False, removeCMMotion=True)
            system = Coulomb.No_Long(system)
            system = Geometric.GeometricMix(system,0.5)
            for i in range(system.getNumForces()):
                force = system.getForce(i)
                force.setForceGroup(i)

            forces = {system.getForce(index).__class__.__name__: system.getForce(
                        index) for index in range(system.getNumForces())}
            
            nonbonded_force = forces['NonbondedForce']
            geometric_nonbonded_force = system.getForce(6)
            coulomb_force = system.getForce(5)

            geometric_nonbonded_force.setUseLongRangeCorrection(False)
            coulomb_force.setUseLongRangeCorrection(False)
            nonbonded_force.setUseDispersionCorrection(False)
            nonbonded_force.setUseSwitchingFunction(False)
            nonbonded_force.setExceptionsUsePeriodicBoundaryConditions(True)

            if no_charge == True:
                for index in range(nonbonded_force.getNumParticles()):
                    charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
                    nonbonded_force.setParticleParameters(index, charge*0, sigma, epsilon)
                    charge = coulomb_force.getParticleParameters(index)
                    coulomb_force.setParticleParameters(index, (0,))
                for i in range(nonbonded_force.getNumExceptions()):
                    (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
                    nonbonded_force.setExceptionParameters(i, p1, p2, q*0, sig, eps)

            if no_epsilon == True:
                for index in range(nonbonded_force.getNumParticles()):
                    charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
                    nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)
                    sigma, epsilon = geometric_nonbonded_force.getParticleParameters(index)
                    geometric_nonbonded_force.setParticleParameters(index, [sigma, epsilon * 0])

                for i in range(nonbonded_force.getNumExceptions()):
                    (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
                    nonbonded_force.setExceptionParameters(i, p1, p2, q, sig, eps*0)

            if no_HB == True:
                for pair in postdeleteHBpairs:
                    nonbonded_force.addException(pair[0], pair[1], 0, 1, 0)
                    geometric_nonbonded_force.addExclusion(pair[0],pair[1])
                    coulomb_force.addExclusion(pair[0],pair[1])

            if hansen_parameter == 'HB':
                for pair in postdeleteHBpairs:
                    sig1,eps1 = geometric_copy.getParticleParameters(int(pair[0])+minnoDelete)
                    sig2,eps2 = geometric_copy.getParticleParameters(int(pair[1])+minnoDelete)
                    charge1 = coulomb_copy.getParticleParameters(int(pair[0])+minnoDelete)[0]
                    charge2 = coulomb_copy.getParticleParameters(int(pair[1])+minnoDelete)[0]
                    sig_mix = np.sqrt(sig1*sig2)
                    eps_mix = np.sqrt(eps1*eps2)
                    charge_mix = charge1*charge2
                    nonbonded_force.addException(pair[0], pair[1], charge_mix, sig_mix, eps_mix)
                    geometric_nonbonded_force.addExclusion(pair[0],pair[1])
                    coulomb_force.addExclusion(pair[0],pair[1])
                    
                    # pair = [i+minnoDelete for i in pair]
                    # sig1,eps11 = geometric_copy.getParticleParameters(int(pair[0]))
                    # sig2,eps22 = geometric_copy.getParticleParameters(int(pair[1]))
                    # charge1, sigma, epsilon = nonbonded_copy.getParticleParameters(int(pair[0]))
                    # charge2, sigma, epsilon = nonbonded_copy.getParticleParameters(int(pair[1]))
                    # charge1 = coulomb_copy.getParticleParameters(int(pair[0]))[0]
                    # charge2 = coulomb_copy.getParticleParameters(int(pair[1]))[0]
                    # sig_mix = np.sqrt(sig1*sig2)
                    # eps_mix = np.sqrt(eps1*eps2)
                    # charge_mix = charge1*charge2
                    # pos1 = np.array([i._value for i in pos_copy[int(pair[0])]])
                    # pos2 = np.array([i._value for i in pos_copy[int(pair[1])]])
                    # pos1 = np.where(pos1<0,load_L+pos1,pos1)
                    # pos2 = np.where(pos2<0,load_L+pos2,pos2)
                    # delta = pos1-pos2
                    # delta = np.where(delta > 0.5*load_L, delta-load_L, np.where(delta < -0.5*load_L, delta+load_L, delta))
                    # r = np.linalg.norm(delta)
                    # ene += (1.60217663e-19**2)*(charge_mix)/(r*1e-9*8.8541878128e-12*4*np.pi)/1000*(6.02214076e23)
                    # ene += 4*eps_mix*((sig_mix/r)**12 - (sig_mix/r)**6)
                    # print((1.60217663e-19**2)*(charge_mix._value)/(r*1e-9*8.8541878128e-12*4*np.pi)/1000*(6.02214076e23) - liquid_energy)
                    # print(4*eps_mix*((sig_mix/r)**12 - (sig_mix/r)**6))
                    # print(pairiter, (1.60217663e-19**2)*(charge_mix)/(r*1e-9*8.8541878128e-12*4*np.pi)/1000*(6.02214076e23), liquid_energy)
                    # print(pairiter, 4*eps_mix*((sig_mix/r)**12 - (sig_mix/r)**6), liquid_energy)
                    # print(liquid_energy)
                #     print(eps11,eps1,eps22,eps2)
                # print(ene)

            integrator = LangevinIntegrator(TEMPERATURE*kelvin,1/picosecond,TIMESTEP*picoseconds)
            simulation = Simulation(topology=modeller.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
            simulation.context.setPositions(pos)
            simulation.context.setVelocities(vel)
            simulation.context.setPeriodicBoxVectors(np.array([gas_box_size,0,0]),np.array([0,gas_box_size,0]),np.array([0,0,gas_box_size]))

            residue_energy = simulation.context.getState(getEnergy=True,groups={3,5,6}).getPotentialEnergy()._value #in kJ/mol
            gas_energies += residue_energy

        deltaE = (gas_energies - liquid_energy) #kJ/mol

        if deltaE < 0:
            multiplier = -1
        else:
            multiplier = 1

        hildebrand = multiplier* np.sqrt(np.abs(deltaE)/system_volume * 1000) #1kJ/cm3 = 1000 MPa, so hildebrand variable is in MPa^0.5
        hansen_dict[hansen_parameter].append(hildebrand)


for hansen_parameter in hansen_parameters:
    f_out.write('{} = {}\n'.format(hansen_parameter,np.mean(hansen_dict[hansen_parameter])))
f_out.close()
