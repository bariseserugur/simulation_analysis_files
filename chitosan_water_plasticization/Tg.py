import numpy as np
from scipy.stats import sem
import random
import matplotlib
import os
import sys
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

#####
concentrations = [0,1,5,10,15,20,30,40]
target_dict = {}
target_dict['glycerol'] = [900 for i in concentrations]
target_dict['water'] = [900,900,900,800,800,700,600,600]
target_dict['hpmcas'] = [900,900,900,850,800]
target_dict['hpmcp'] = [900,900,900,850,800]
target_dict['pvp'] = [900,900,900,850,800]
target_dict['pvpva'] = [900,900,900,850,800]
target_dict['pmma'] = [900,900,900,850,800]
#####

##### Input parameters
SOLVENT = str(sys.argv[1])

def get_temperature_range(solvent,composition):
    comp_ix = concentrations.index(composition)
    initial_t = target_dict[solvent][comp_ix]
    temperature_range = np.arange(initial_t,199,-1)
    return temperature_range

def broken_stick(T,rho):
    # print(list(rho))
    l1range = [max(T)-200,max(T)-10]
    l2range = [min(T)+10,min(T)+200]
    l1ixrange = [index for index,obj in enumerate(T) if l1range[0]<obj<l1range[1]]
    l2ixrange = [index for index,obj in enumerate(T) if l2range[0]<obj<l2range[1]]
    Tglist = []
    while len(Tglist) <= 1000:
        randoml1 = random.sample(l1ixrange,1)[0] #index of the end points for slope
        randoml2 = random.sample(l2ixrange,1)[0] #index of the end points for slope
        if len(T[:randoml1]) < 5 and len(T[randoml2:]) < 5:
            continue
        l1endT = T[randoml1] #end temperature of slope line
        l2endT = T[randoml2] #end temperature of slope line
        line1fit = np.poly1d(np.polyfit(T[:randoml1], rho[:randoml1], 1))
        line2fit = np.poly1d(np.polyfit(T[randoml2:], rho[randoml2:], 1))
        slope1,yint1=line1fit[1],line1fit[0]
        slope2,yint2=line2fit[1],line2fit[0]
        int_x = (yint2-yint1)/(slope1-slope2)
        if l1endT>int_x>l2endT:
            Tglist.append(int_x)
    # print(len(np.unique(Tglist)))
    if 1 in [1]: #COMPOSITION ==0:
        plt.scatter(T,rho,color='blue',label='Conc: {}, Tg= {:.1f} K'.format(COMPOSITION,np.mean(Tglist)))
        plt.plot(T,line1fit(T),color='red')
        plt.plot(T,line2fit(T),color='red')
        plt.axvline(np.mean(Tglist),linestyle='--')
        plt.legend()
        plt.show()
    return np.mean(Tglist), np.std(Tglist)

def broken_stick(T,rho):
    Tglist = []
    while len(Tglist) <= 1000:
        randoml1 = random.sample(list(np.arange(5,16,1)),1)[0]
        randoml2 = random.sample(list(np.arange(-16,-4,1)),1)[0]
        line1fit = np.poly1d(np.polyfit([T[0],T[randoml1]], [rho[0],rho[randoml1]], 1))
        line2fit = np.poly1d(np.polyfit([T[randoml2],T[-1]], [rho[randoml2],rho[-1]], 1))
        slope1,yint1=line1fit[1],line1fit[0]
        slope2,yint2=line2fit[1],line2fit[0]
        int_x = (yint2-yint1)/(slope1-slope2)
        if T[randoml1]>int_x>T[randoml2]:
            Tglist.append(int_x)
        if COMPOSITION == 0 and simno==1:
            print(T[randoml1],T[randoml2])
    return np.mean(Tglist), np.std(Tglist)

# def broken_stick(T,rho):
#     Tglist = []
#     while len(Tglist) <= 1000:
#         randoml1 = random.sample(list(np.arange(0,13,1)),1)[0]
#         randoml2 = random.sample(list(np.arange(len(T)-16,len(T)-3,1)),1)[0]
#         line1fit = np.poly1d(np.polyfit(T[randoml1:randoml1+3], rho[randoml1:randoml1+3], 1))
#         line2fit = np.poly1d(np.polyfit(T[randoml2:randoml2+3], rho[randoml2:randoml2+3], 1))
#         l1endT,l2endT = max(T[randoml1:randoml1+3]), min(T[randoml2:randoml2+3])

#         slope1,yint1=line1fit[1],line1fit[0]
#         slope2,yint2=line2fit[1],line2fit[0]
#         int_x = (yint2-yint1)/(slope1-slope2)
#         if l1endT>int_x>l2endT:
#             Tglist.append(int_x)
#     return np.mean(Tglist), np.std(Tglist)

def get_density(thermofile,temperature_range):
    thermofile.readline()
    density_list = []
    for temperature in temperature_range:
        if temperature % 10 == 0:
            temp_density = []
            for iter in range(1000):
                thermofile.readline()
            for iter in range(1000):
                line = thermofile.readline()
                density = float(line.split(',')[-1])
                temp_density.append(density)
            density_list.append(np.mean(temp_density))
        else:
            for iter in range(100):
                thermofile.readline() #discard every 1K
    return density_list
  
#every 10K method
if SOLVENT in ['water','glycerol']:
    concentrations = [0,1,5,10,15,20,30,40]
    simnos = [1,2,3,4]
    water_glycerol = True
else:
    concentrations = [100,80,60,40,20]
    simnos = [1]
    water_glycerol = False

all_Tgs = []
all_sems = []
all_stds = []
for COMPOSITION in concentrations:
    temp_range = get_temperature_range(SOLVENT,COMPOSITION)
    INITIAL_T = temp_range[0]
    Tglist = []
    stdlist = []
    for simno in simnos:
        if water_glycerol == True:
            thermofile = open('/scratch/gpfs/bu9134/chitosan_{}_900_200/{}/{}/{}_200.avg'.format(SOLVENT,COMPOSITION,simno,INITIAL_T),'r')
        else:
            thermofile = open('/scratch/gpfs/bu9134/stephanie_CBD/{}_OpenMM/{}/{}_200.avg'.format(SOLVENT.upper(),COMPOSITION,INITIAL_T),'r')
        rho = get_density(thermofile,temp_range)
        temperatures = np.linspace(max(temp_range),min(temp_range),len(rho))
        Tg,std = broken_stick(temperatures,rho)
        if COMPOSITION == 0:
            print(rho)
        Tglist.append(Tg)
        stdlist.append(std)
    all_Tgs.append(np.mean(Tglist))
    all_stds.append(np.mean(stdlist))
    all_sems.append(sem(Tglist))
# all_Tgs.reverse()
# all_sems.reverse()
print(all_Tgs)
print(all_stds)
print(all_sems)



# ##every 1K method
# for comp in [40]:#[0,1,5,10,15,20,30,40]:
#     Tglist = []
#     for simno in [1,2,3,4]:
#         f = open('/scratch/gpfs/bu9134/Plasticization/chitosan/plasticization/{}/{}/thermo.avg'.format(comp,simno),'r')
#         f2 = open('/scratch/gpfs/bu9134/Plasticization/chitosan/plasticization/{}/{}/thermo.avg'.format(comp,simno),'r')
#         filelen = len(f2.readlines()[1:])
        
#         f.readline()
#         for x in range(37425*2):
#             f.readline() 
#         rho = []
#         for x in range((filelen-37425*2)//2000):
#             for i in range(10):
#                 minirho = []
#                 for x in range(200):
#                     line = f.readline()
#                     if len(line) != 0:
#                         minirho.append(float(line.split(',')[-1]))
#                 rho.append(np.mean(minirho))
#         #print(rho)
#         #print(len(rho))
#         t = list(range(500,500-len(rho),-1))
#         Tg = broken_stick(t,rho)
#         if np.isnan(Tg) == False:
#             Tglist.append(Tg)
#     print(comp,Tglist,np.mean(Tglist),sem(Tglist))
