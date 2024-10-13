import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import argparse
import sys
import time
import os
import pickle


free_volume_fraction = 0
trajct = 0
for traj in np.arange(20,100020,10000):
    f = open('free_volume_outputs/0.2_{}.txt'.format(traj), 'rb')
    grid = pickle.load(f)
    total_grid_size = len(grid)
    grid = grid[grid[:,3]==0][:,:3]
    free_volume_fraction += len(grid)/total_grid_size
    trajct+=1
free_volume_fraction = free_volume_fraction/trajct
f_out = open('fv_fraction.out','w')
f_out.write('fvf = {}\n'.format(free_volume_fraction))
