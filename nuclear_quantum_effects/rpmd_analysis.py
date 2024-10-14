import time as time
from pymbar import timeseries
import itertools
import os
import numpy as np
import sys
import ast

'''Compile and compute the potential, kinetic, and total energies, temperatures, volumes, densities and additional parameters for each configuration simulated via path-integral MD (PIMD).'''

kb = 8.31446261815324 #kJ/molK
avogadro = 6.0221409e23
atm_to_pa = 101325 # atm to pascals
pressure = 1 #atm
PV_const = pressure * 1E-27 * atm_to_pa * avogadro / 1000  # nm^3 to m^3, atm to Pa, J to kJ, per system to per mol
Jtohartree = 2.2937104486906e17

dirname = sys.argv[1]

analysis_dir = "RPMD_TAFFI_Results"
output_dict_name = 'rpmd_thermo'


scratch_path = '/scratch/gpfs/bu9134'
path = scratch_path + '/' + analysis_dir
dielectric_path = path + '/dielectric_outputs'

molecules = [dirname for dirname in next(os.walk(path))[1] if (os.path.isdir(path+'/{}/1'.format(dirname)))]
sim_no_list = [1,2,3,4]

processed_data = open(path+'/processed_data/{}.data'.format(dirname),'w')
P = 32
rpmd_dipoles_dict = dict()
rpmd_M2_dict = dict()

parameter_names = ['PE','KE','PEtot','TE','T','volume','density','dE2','dHdV','dV2','dipole','M2']
def thermo_analysis(molecule):
    thermo = dict()
    four_sim_thermo = dict()
    pymbar_thermo = dict()
    
    for parameter_name in parameter_names:
        four_sim_thermo[parameter_name] = []
        pymbar_thermo[parameter_name] = []

    for simno in sim_no_list:
        total_linecount = 0
        linecounts = []
        if os.path.isfile(path+'/{}/{}/thermo4_1.avg'.format(molecule,simno)):
            files = ['/scratch/gpfs/bu9134/RPMD_TAFFI_Results/{}/{}/thermo4_1.avg'.format(molecule,simno),path+'/{}/{}/thermo4_2.avg'.format(molecule,simno),path+'/{}/{}/thermo4_3.avg'.format(molecule,simno),path+'/{}/{}/thermo4_4.avg'.format(molecule,simno)]
            for i in range(len(files)):
                linecount = int(os.popen('wc -l /scratch/gpfs/bu9134/RPMD_TAFFI_Results/"{}"/{}/thermo4_{}.avg'.format(dirname,simno,i+1)).read().split()[0])
                linecounts.append(linecount)
                total_linecount += linecount
        elif os.path.isfile(path+'/{}/{}/thermo2_1.avg'.format(molecule,simno)):
            files = ['/scratch/gpfs/bu9134/RPMD_TAFFI_Results/{}/{}/thermo2_1.avg'.format(molecule,simno),path+'/{}/{}/thermo2_2.avg'.format(molecule,simno)]
            for i in range(len(files)):
                linecount = int(os.popen('wc -l /scratch/gpfs/bu9134/RPMD_TAFFI_Results/"{}"/{}/thermo2_{}.avg'.format(dirname,simno,i+1)).read().split()[0])
                linecounts.append(linecount)
                total_linecount += linecount
        elif os.path.isfile('/scratch/gpfs/bu9134/RPMD_TAFFI_Results/{}/{}/thermo.avg'.format(molecule,simno)):
            files = [path+'/{}/{}/thermo.avg'.format(molecule,simno)]
            linecount = int(os.popen('wc -l /scratch/gpfs/bu9134/RPMD_TAFFI_Results/"{}"/{}/thermo.avg'.format(dirname,simno)).read().split()[0])
            linecounts.append(linecount)
            total_linecount += linecount
        total_linecount = total_linecount - len(files)
        thermo_array = np.zeros((total_linecount,8))
        counter = 0
        for index,obj in enumerate(files):
            with open(obj,'r') as thermo_file:
                next(thermo_file)
                for line_iter in range(linecounts[index]-1):
                    line = thermo_file.readline()
                    thermo_array[counter] = np.array([float(x) for x in line.split(',')])
                    counter+=1
        QKEline = open('/scratch/gpfs/bu9134/RPMD_TAFFI_Results/{}/{}/QKE.data'.format(molecule,simno),'r').readlines()[0]
        QKE = np.array(ast.literal_eval(QKEline[6:]))
        
        step = thermo_array[:,0]-6000000
        time = (thermo_array[:,1]-3000)/1000 #nanoseconds

        thermo['PE'] = np.array([np.sum(thermo_array[:,2][i:i+P])/P for i in np.arange(0,total_linecount,P)])[9::10] #kJ/mol
        thermo['KE'] = np.array([np.sum(thermo_array[:,3][i:i+P])/P for i in np.arange(0,total_linecount,P)])[9::10] #kJ/mol
        thermo['PEtot'] = np.array(QKE/P)
        thermo['TE'] = thermo['PEtot'] + thermo['KE']
        thermo['T'] = thermo_array[:,5] #K
        thermo['T'] = [thermo['T'][index*32:index*32+32] for index,obj in enumerate(thermo['T'][9::10])]
        thermo['T'] = np.array([item for sublist in thermo['T'] for item in sublist])
        thermo['volume'] = np.array([thermo_array[:,6][i] for i in np.arange(0,total_linecount,P)])[9::10] #nm^3
        thermo['density'] = np.array([thermo_array[:,7][i] for i in np.arange(0,total_linecount,P)])[9::10] #g/cm3
        thermo['dE2'] = (thermo['TE'] - np.mean(thermo['TE']))**2
        H = thermo['PEtot'] + thermo['volume']*PV_const
        dH = H - np.mean(H)
        avgV = np.mean(thermo['volume'])
        dV = thermo['volume'] - avgV
        thermo['dHdV'] = dH*dV
        thermo['dV2'] = dV**2
        dipole_file = open(dielectric_path+'/{}_{}.di'.format(molecule,simno))
        dipole_lines = dipole_file.readlines()
        for dipole_line in dipole_lines:
            exec(dipole_line,None,globals())
        thermo['dipole'] = np.array(rpmd_dipoles_dict[molecule,simno])
        thermo['M2'] = np.array(rpmd_M2_dict[molecule,simno])
        
        for parameter_name in parameter_names:
            if parameter_name == 'dipole':
                four_sim_thermo[parameter_name].append(list(np.mean(thermo[parameter_name],axis=0)))
            else:
                four_sim_thermo[parameter_name].append(np.mean(thermo[parameter_name]))

        for parameter_name in parameter_names:
            if parameter_name not in ['T','density','dipole']:
                [t0, g, Neff_max] = timeseries.detectEquilibration(thermo[parameter_name])
                A_t_equil = thermo[parameter_name][t0:]
                indices = timeseries.subsampleCorrelatedData(A_t_equil,g=g)
                absolute_indices = [x+t0 for x in indices]
                pymbar_thermo[parameter_name].extend(list(thermo[parameter_name][absolute_indices]))
                if parameter_name == 'volume':
                    pymbar_thermo['density'].extend(list(thermo['density'][absolute_indices]))
                elif parameter_name == 'KE':
                    pymbar_thermo['T'].extend(list(thermo['T'][absolute_indices]))
                elif parameter_name == 'M2':
                    pymbar_thermo['dipole'].extend(list([list(x) for x in thermo['dipole'][absolute_indices]]))
    return four_sim_thermo,pymbar_thermo

four_sim_thermo,pymbar_thermo = thermo_analysis(dirname)

for parameter_name in parameter_names:
    processed_data.write('{}_four_sim["{}","{}"] = {}\n'.format(output_dict_name,dirname,parameter_name,four_sim_thermo[parameter_name]))
    processed_data.write('{}_pymbar["{}","{}"] = {}\n'.format(output_dict_name,dirname,parameter_name,pymbar_thermo[parameter_name]))
