import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import argparse
import sys
import time
import os
import pickle
import shutil

'''This code yields a grid array (N,4) that indicates the location of the grid, and whether it is occupied or not.
It runs in parallel using 96 cores via joblib.'''

def create_parser():
    parser = argparse.ArgumentParser(description='Identify points of no overlap (unoccupied volume)')
    parser.add_argument('traj',
                        help='trajectory file')
    parser.add_argument('-rprobe', default=0.2, help='Size of probe. (default=0.1)')
    parser.add_argument('-samp_freq', default=1, help='Frequency (in terms of frames) to sample configurations/data. (default=100)')
    parser.add_argument('-max_samp', default=10, help='Maximum number of frames to sample from the trajectory file. (default=10)')
    parser.add_argument('-nskip', default=0, help='Number of frames to skip in trajectory file. (default=5000)')
    parser.add_argument('-output', default='freevolume', help='output file name (default=probes)')
    return parser

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def find_overlaps(r_probe, grid, all_atom_info, box_info): 
    # calculate distance from polymer molecules
    # if distance is < r_probe + r_polymer: overlaps
    overlaps = np.zeros(len(grid))
    l = 0
    for coord in grid: 
        # calculates dist_pbc between probe coord and all atom positions
        # then checks if any of these distances is less than r_probe+r_atom
        # flips overlaps[probe idx] if any of these distances is.
        dists = dist_pbc(all_atom_info['xs'], coord, box_info['length'])**0.5 < (r_probe + all_atom_info['sigmas']/2)
        if np.any(dists): 
            overlaps[l] = 1
        l += 1
    return overlaps

def main(argv): 
    start_time = time.time()
    parser = create_parser()
    args = parser.parse_args()
    traj_file = args.traj
    output_file = args.output
    r_probe = float(args.rprobe)
    nskip = int(args.nskip)
    max_samp = int(args.max_samp)
    samp_freq = int(args.samp_freq)

    target_directory = os.getcwd()+'/'
    output_directory = target_directory+'free_volume_outputs/'

    if os.path.exists(output_directory) == False:
        os.makedirs(output_directory)
 
    #Read sys.pdb file
    atom_names = []
    fid = open('sys.pdb','r').readlines()
    for line in fid:
        if 'ATOM' in line:
            atom_names.append(line.split()[2])
        if 'CONECT' in line:
            break
    
    #Read ff_sys.xml file
    name_to_type = {}
    type_to_sigma = {}
    fid = open('ff_sys.xml','r').readlines()
    for line in fid:
        if '<Atom name=' in line:
            name_to_type[line.split('"')[1]] = line.split('"')[3]
        if '<Atom type=' in line:
            type_to_sigma[line.split('"')[1]] = 10*float(line.split('"')[5]) #in Angstroms
    atom_sigmas = np.array([type_to_sigma[name_to_type[i]] for i in atom_names])
    atom_count = len(atom_sigmas)

    # process trajectory file & make output files
    print("Reading %s" % traj_file)
    print("Skipping %d frames" % nskip)
    print("Reading %d frames with a sampling frequency of %d" % (max_samp, samp_freq))
    fid = open(traj_file)
    frame = 0
    nlog = 0
    still_data = fid.readline()
    stillLog = True
    while still_data and stillLog: 
        if frame >= nskip and (frame-nskip) % samp_freq == 0:
            box_info = {'size': [[], [], []], 
                        'length': np.zeros(3)}
            timestep = int(fid.readline())
            print("Reading timestep %d" % timestep)

            for i in range(3): 
                fid.readline()
            for i in range(3): 
                line = fid.readline().strip().split()
                box_info['size'][i] = [float(line[0]), float(line[1])]
                box_info['length'][i] = float(line[1]) - float(line[0])
            fid.readline()

            all_atom_info = {
                            'ids': [0]*atom_count, 
                            'xs': np.zeros((atom_count, 3)),
                            'sigmas': np.zeros(atom_count)
                            }
            
            #fill in atom information
            for aa in range(atom_count): 
                line = fid.readline().strip().split()
                all_atom_info['ids'][aa] = int(line[0])
                all_atom_info['xs'][aa][0] = float(line[-3])
                all_atom_info['xs'][aa][1] = float(line[-2])
                all_atom_info['xs'][aa][2] = float(line[-1])
            all_atom_info['sigmas'] = atom_sigmas

            # generate grid for placing probers on
            gridx = np.arange(box_info['size'][0][0], box_info['size'][0][1], 2*r_probe)
            gridy = np.arange(box_info['size'][1][0], box_info['size'][1][1], 2*r_probe)
            gridz = np.arange(box_info['size'][2][0], box_info['size'][2][1], 2*r_probe)
            ndim = 3
            grid = np.array(np.meshgrid(gridx,gridy,gridz)).T.reshape(-1, ndim)
            
            def split(a, n):
                k, m = divmod(len(a), n)
                return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
            grids = split(grid,96)
 
            parallel_outputs = Parallel(n_jobs=96)(delayed(find_overlaps)(r_probe,grids[i],all_atom_info,box_info) for i in range(96))
            all_overlaps = np.hstack(parallel_outputs)
            
            output_array = np.zeros((len(all_overlaps),4)) 
            for index,obj in enumerate(all_overlaps):
                output_array[index] = np.hstack((grid[index],np.array([obj])))
 
            output_name = output_file + "_{}.txt".format(timestep)
            f = open(output_directory+output_name, "wb")
            f.close()

            f = open(output_directory+output_name, "ab")
            pickle.dump(output_array, f)
            f.close()
            nlog +=1 

        else: 
            # skip frame
            for i in range(atom_count + 8): # header is 9 lines long but first line is read by still_data
                fid.readline()
        
        frame +=1 
        if nlog > max_samp: 
            stillLog = False
        still_data = fid.readline()

    print("total time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__": 
    main(sys.argv[1:])