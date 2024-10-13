import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import argparse
import sys
import time
import os
import pickle

def create_parser():
    parser = argparse.ArgumentParser(description='Identify points of no overlap (unoccupied volume)')
    parser.add_argument('traj',
                        help='trajectory file')
    parser.add_argument('-rprobe', default=0.1, help='Size of probe. (default=0.1)')
    parser.add_argument('-samp_freq', default=100, help='Frequency (in terms of frames) to sample configurations/data. (default=100)')
    parser.add_argument('-max_samp', default=1, help='Maximum number of frames to sample from the trajectory file. (default=10)')
    parser.add_argument('-nskip', default=0, help='Number of frames to skip in trajectory file. (default=5000)')
    parser.add_argument('-output', default='probes', help='output file name (default=probes)')
    return parser

def dist_pbc(x0, x1, dimensions): 
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def find_proximity(grid, all_polymer_info, box_info, proximity): 
    # calculate distance from polymer molecules
    # if distance is < 5 A: within proximity
    free_neighs = 0
    for coord in grid: 
        # calculates dist_pbc between probe coord and polymer atom positions
        # then checks if any of these distances is less than r_probe+r_atom
        dists = dist_pbc(all_polymer_info['xs'], coord, box_info['length'])**0.5 < proximity
        if np.any(dists):
            free_neighs += 1
    return free_neighs

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
    read_directory = target_directory+'free_volume_outputs/'

    #Read sys.pdb file
    atom_names = []
    fid = open('sys_evaporated.pdb','r').readlines()
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
            type_to_sigma[line.split('"')[1]] = 10*float(line.split('"')[5])
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
                            'ids': np.array([0]*atom_count), 
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
            all_atom_info['sigmas'] = np.array(atom_sigmas)

            #load previously analyzed grid (occupied/unoccupied)
            
            f_out = open('segment_' + "_%d.txt".format(traj_file) % timestep, "w")

            gridlist = []

            for core_number in range(96):
                read_name = output_file + "_{}_{}.txt".format(timestep,core_number)
                f_read = open(read_directory+read_name, "rb")
                gridlist.append(pickle.load(f_read))

            grid = np.vstack(gridlist)
                       
            #filter grid only to display unoccupied grids
            grid = grid[grid[:,3]==0][:,:3]
            
            # def split(a, n):
            #     k, m = divmod(len(a), n)
            #     return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

            # grid = split(grid,96)
            # grid = grid[int(core_number)]

            #split info into polymers 
            #the below section is for per polymer free volume analysis. Alternatively, will do per polymer segment analysis
            # polymer_count = 10
            # polymer_indices = [[] for i in range(polymer_count)]
            # sysf = open('sys_evaporated.pdb','r').readlines()
            # for line in sysf:
            #     split = line.split()
            #     if 'AAA' in line and split[-1] != 'H':
            #         atomindex = int(split[1])-1
            #         moleculeindex = int(split[4])-1
            #         polymer_indices[moleculeindex].append(atomindex)


            #per polymer segment analysis
            one_polymer_segments = [np.array(i) for i in [range(222),range(222,442),range(442,662),range(662,882),range(882,1103)]]
            all_polymer_segments = []
            for poly_no in range(10):
                all_polymer_segments.extend([i+poly_no*1103 for i in one_polymer_segments])
            polymer_indices = all_polymer_segments #to keep the naming consistent with the per-polymer analysis, rename the polymer segment indices

            free_neigh_counts = [0 for i in polymer_indices] 
            for polymer_no, polyixlist in enumerate(polymer_indices):
                print(polymer_no)
                all_polymer_info = {}
                for key,value in all_atom_info.items():
                    all_polymer_info[key] = value[polyixlist]
                free_neigh_counts[polymer_no] += find_proximity(grid, all_polymer_info, box_info, 5) #proximity criteria of 5 Angstroms
            
            f = open('segment_fv' + "_%d.txt".format(traj_file) % timestep, "a")
            f.write('{}\n'.format(free_neigh_counts))
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