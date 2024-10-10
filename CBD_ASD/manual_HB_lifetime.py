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
from scipy.optimize import curve_fit

'''This code analyzes the HB lifetimes for polymer-polymer, polymer-solvent, and solvent-solvent hydrogen bonds. 
The output consists of five numpy saves: 
1) the array of tau_values, 
2) polymer-polymer HB lifetime array, 
3) polymer_acceptor-solvent_donor HB lifetime array, 
4) polymer_donor-solvent_acceptor HB lifetime array, 
5) solvent-solvent HB lifetime array.'''

sttime = time.time()

POLY = sys.argv[1]

topology = md.load('sys.pdb').topology
total_atom_count = topology.n_atoms


if topology.atom(total_atom_count-1).residue.n_atoms != 53:
    last_CBD_index = total_atom_count #atom with this index doesn't exist, this deals with the scenarios for 100% polymer systems
    no_cbd_check = True
else:
    CBD_residue_name = topology.atom(total_atom_count-1).residue.name
    last_CBD_index = min([i.index for i in topology.atoms if i.residue.name == CBD_residue_name])
    no_cbd_check = False


f = open('HB_310_dynamics.out', 'rb')
HB_array = pickle.load(f)
HB_array_og = copy.deepcopy(HB_array)


frames = np.arange(0,10000,1)

def HB_lifetime(input_HB_array, start_frames):
    all_masks = []
    input_HB_array_copy = copy.deepcopy(input_HB_array)

    for start_frame in start_frames:
        first_frame_HBs = []

        input_HB_array = input_HB_array_copy[(input_HB_array_copy[:,0]>=start_frame) & (input_HB_array_copy[:,0]<start_frame+WINDOW_SIZE)]

        HB_start_frame = input_HB_array[input_HB_array[:,0]==start_frame][:,[1,3]]

        for row in HB_start_frame:
            first_frame_HBs.append(row)
        
        remaining_frames = frames[start_frame:start_frame+WINDOW_SIZE]

        HB_frame = input_HB_array[:,[0,1,3]]

        for HBix,HB in enumerate(first_frame_HBs):
            whereequal = (HB_frame[:,[1,2]]==HB)
            HBpresence_frames = HB_frame[np.all(whereequal,axis=1)][:,0]
            
            mask = np.isin(remaining_frames, HBpresence_frames)
            presences_binary = mask.astype(int)
            # consecutive_zeros = np.convolve(presences_binary, np.ones(INTERMITTENCY), 'valid') == 0

            # if np.any(consecutive_zeros) == False:
            #     #mask = np.ones(WINDOW_SIZE)
            #     continue
            # else:
            #     break_frame = remaining_frames[np.where(consecutive_zeros == True)][0]
            #     mask[break_frame - start_frame:] = 0
            
            mask = mask.astype(int)
            all_masks.append(mask)
    return all_masks


def fit_biexponential(tau_timeseries, ac_timeseries):
    """Fit a biexponential function to a hydrogen bond time autocorrelation function
    Return the two time constants
    """
    def model(t, A, tau1, B, tau2):
        """Fit data to a biexponential function.
        """
        return A * np.exp(-t / tau1) + B * np.exp(-t / tau2)

    params, params_covariance = curve_fit(model, tau_timeseries, ac_timeseries, [1, 0.5, 1, 2])

    fit_t = np.linspace(tau_timeseries[0], tau_timeseries[-1], len(tau_timeseries))
    fit_ac = model(fit_t, *params)

    return params, fit_t, fit_ac


f_out = open('manual_hb_lifetimes.out','w')
WINDOW_SIZE = 5000
WINDOW_STEP = 250
INTERMITTENCY = 4999
start_frames = np.arange(0,len(frames)-WINDOW_SIZE+WINDOW_STEP,WINDOW_STEP)

from scipy.signal import savgol_filter   
import pandas as pd

if no_cbd_check == False:
    #CBD-CBD
    HB_array = HB_array_og[(HB_array_og[:,1] >= last_CBD_index) & (HB_array_og[:,3] >= last_CBD_index)]
    all_masks = HB_lifetime(HB_array, start_frames)

    mask = np.mean(all_masks,axis=0)
    mask = list(mask)
    # df = pd.DataFrame(dict(x=mask))
    # mask = df[["x"]].apply(savgol_filter,  window_length=301, polyorder=2)

    params, fit_t, fit_ac = fit_biexponential(np.linspace(0,WINDOW_SIZE,len(mask)), mask)
    # print(list(fit_ac))
    A, tau1, B, tau2 = params
    time_constant = A * tau1 + B * tau2
    time_constant = np.mean(mask[1000:5000])
    f_out.write('cbd_cbd = {}\n'.format(time_constant))

    #CBD-Polymer
    HB_array1 = HB_array_og[(HB_array_og[:,1] >= last_CBD_index) & (HB_array_og[:,3] < last_CBD_index)]
    HB_array2 = HB_array_og[(HB_array_og[:,1] < last_CBD_index) & (HB_array_og[:,3] >= last_CBD_index)]
    HB_array = np.vstack([HB_array1,HB_array2])

    all_masks = HB_lifetime(HB_array, start_frames)
    mask = np.mean(all_masks,axis=0)
    mask = list(mask)

    params, fit_t, fit_ac = fit_biexponential(np.linspace(0,WINDOW_SIZE,len(mask)), mask)
    A, tau1, B, tau2 = params
    time_constant = A * tau1 + B * tau2
    time_constant = np.mean(mask[1000:5000])
    f_out.write('poly_cbd = {}\n'.format(time_constant))

    # #Polymer-CBD
    # if POLY.startswith('HPMC'):
    #     HB_array = HB_array_og[(HB_array_og[:,1] < last_CBD_index) & (HB_array_og[:,3] >= last_CBD_index)]
    #     all_masks = HB_lifetime(HB_array, start_frames)
    #     mask = np.mean(all_masks,axis=0)
    #     mask = list(mask)

    #     params, fit_t, fit_ac = fit_biexponential(np.linspace(0,WINDOW_SIZE,len(mask)), mask)
    #     A, tau1, B, tau2 = params
    #     time_constant = A * tau1 + B * tau2
    #     f_out.write('poly_cbd = {}\n'.format(time_constant))

#Polymer-Polymer
if POLY.startswith('HPMC'):
    HB_array = HB_array_og[(HB_array_og[:,1] < last_CBD_index) & (HB_array_og[:,3] < last_CBD_index)]
    all_masks = HB_lifetime(HB_array, start_frames)
    mask = np.mean(all_masks,axis=0)
    mask = list(mask)
    print(mask)

    params, fit_t, fit_ac = fit_biexponential(np.linspace(0,WINDOW_SIZE,len(mask)), mask)
    A, tau1, B, tau2 = params
    time_constant = A * tau1 + B * tau2
    time_constant = np.mean(mask[1000:5000])
    f_out.write('poly_poly = {}\n'.format(time_constant))
f_out.close()