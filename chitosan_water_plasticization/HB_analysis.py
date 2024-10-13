import numpy as np
import sys
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import pickle
import joblib
from joblib import Parallel,delayed

traj_file = sys.argv[1] #trajectory file
pdb_file = sys.argv[2] #.pdb file

u = mda.Universe('modified_{}.pdb'.format(pdb_file.split('.')[0]),'modified_{}.lammpstrj'.format(traj_file.split('.')[0]), format='LAMMPSDUMP', dt=0.01)

def HB_analysis(traj_file,pdb_file,u,start,stop,CUTOFF_DISTANCE = 3.5,CUTOFF_ANGLE = 150):
    
    HB = HydrogenBondAnalysis(
    universe=u,
    donors_sel='element O or element N',
    hydrogens_sel='element H',
    acceptors_sel='element O or element N',
    d_a_cutoff=CUTOFF_DISTANCE,
    d_h_a_angle_cutoff=CUTOFF_ANGLE,
    update_selections=False)

    HB.run(
    start=start,
    stop=stop,
    step=None,
    verbose=True)

    HB_results = HB.results.hbonds
    return HB_results


# HB_results = HB_analysis(traj_file,pdb_file,u,start,stop)
CORE_COUNT = 48
CORE_STEP_SIZE = u.trajectory.n_frames/CORE_COUNT
outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(HB_analysis)(traj_file,pdb_file,u,int(i*CORE_STEP_SIZE),int((i+1)*CORE_STEP_SIZE)) for i in range(CORE_COUNT))
HB_results = np.vstack(outputs)

f = open('HB_{}.out'.format(traj_file.split('.')[0]), 'wb')
pickle.dump(HB_results, f, protocol=4)
f.close()