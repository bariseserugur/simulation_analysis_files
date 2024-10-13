import numpy as np
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import argparse
import sys
import time
import os
import pickle
import numpy as np
import networkx as nx
import sklearn.feature_extraction.image

R_PROBE = float(sys.argv[1])

for traj in [0]: #np.arange(20,100020,10000):
    f = open('fvf{}.out'.format(R_PROBE),'rb')
    fvf = np.load(f)
    #load and modify the grids to return the grid order/number instead of coordinate
    empty_grids = np.array(np.load(f)) / (R_PROBE*2)
    filled_grids = np.array(np.load(f)) 
    #only get non-repeating elements
    filled_grids = np.unique(filled_grids,axis=0)
    filled_grids = filled_grids / (R_PROBE*2)

    empty_grids = np.hstack((empty_grids,np.ones(len(empty_grids)).reshape(-1,1))) #mark empty grids as 1 in the 4th column
    filled_grids = np.hstack((filled_grids,np.zeros(len(filled_grids)).reshape(-1,1))) #mark filled grids as 0 in the 4th column
    all_grids = np.vstack((empty_grids,filled_grids))

    grid = np.around(all_grids).astype(int)

    rows = np.array(sorted(list(set(grid[:,0]))))
    row_ct = len(rows)

    arrays = []
    for row in rows:
        flatarray = np.zeros((row_ct,row_ct))
        filtered_data = grid[grid[:,0]==row]
        for filtered_row in filtered_data:
            flatarray[filtered_row[1]][filtered_row[2]] = filtered_row[3]
        flatarray = np.around(flatarray).astype(int)
        arrays.append(flatarray)
    array3d = np.array(arrays)

    print('Array generated!')
    G = nx.Graph()
    print('finding edges...')
    graph = sklearn.feature_extraction.image.grid_to_graph(row_ct, row_ct, row_ct, mask=array3d)
    print('buidling graph...')
    G.add_edges_from(np.vstack([graph.col, graph.row]).T)
    print('finding subgraphs...')
    subgraphs = nx.connected_components(G)
    sorted_subgraphs = sorted(subgraphs, key=len, reverse=True)
    sorted_subgraphs = [i for i in sorted_subgraphs if len(i)>1]

free_volume_distributions = [len(i)*(R_PROBE*2)**3 for i in sorted_subgraphs]
freqs,vals = np.histogram(free_volume_distributions)#,bins=list(np.arange(0,100,0.5)))

freqs = list(freqs)
vals = list(vals)

f_out = open('free_volume_distribution{}.out'.format(R_PROBE),'w')
f_out.write('free_volume_distributions = {}\n'.format(free_volume_distributions))
f_out.write('freqs = {}\n'.format(freqs))
f_out.write('vals = {}\n'.format(vals))
f_out.write('total_free_volume = {}\n'.format(len(empty_grids) * (R_PROBE*2)**3))
# f_out.write('largest_void = {}\n'.format(len(sorted_subgraphs[0])*(0.2**3)))