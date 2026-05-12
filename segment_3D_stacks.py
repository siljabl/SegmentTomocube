
"""
Author S. B. Låstad

Create 3D mask of epithelial monolayers from probability field predicted by catBioM_MlM
"""

import os
import pickle
import argparse
import tifffile
import numpy as np

from pathlib import Path
from datetime import datetime
from skimage.filters import median

from src.Segment3D import *
from src.ImUtils   import commonStackReader


parser = argparse.ArgumentParser(description='Segment cell from MlM probabilities')
parser.add_argument('dir',       type=str, help="path to main dir")
parser.add_argument('--r_min',  type=int, help="min radius of kernel", default='3')
parser.add_argument('--r_max',  type=int, help="max radius of kernel", default='11')
parser.add_argument('--Nframes', type=int, help="Number of frames")
args = parser.parse_args()


# create folders
in_dir = f"{args.dir}predictions/"
assert os.path.exists(in_dir), "In path does not exist, maybe because of missing '/' at the end of path"
print(in_dir)

path = Path(in_dir)
out_dir  = f"{Path(in_dir).parent}{os.sep}segmentations"
mhds_dir = f"{out_dir}{os.sep}mhds"


try:
    os.mkdir(out_dir)
    os.mkdir(mhds_dir)
except:
    pass


# get experiment specific dataseries names
experiment = []
for file in path.glob("*HT3D_0_prob.npy"):
    experiment.append(file.stem.split("HT3D_0_prob")[0])


# create arrays for prediction and filtering
thresholds = np.linspace(0.5, 1, 20, endpoint=False)
kernel = generate_kernel(args.r_min, args.r_max)



# saving MlM threshold of each file
new_log = 1
logfile = f"{mhds_dir}{os.sep}log.txt"
if os.path.exists(logfile):
    new_log = 0



with open(logfile, "a") as log:
    if new_log:
        log.write("# file, zero-level, MlM threshold, r_min, r_max, r2\n")
    log.write(f"date: {datetime.today().strftime('%Y-%m-%d')}\n")

    # sort by experiment
    for exp in experiment:
        print(exp)

        n_z_list   = []
        dn_dz_list = []

        # compute list for determination of zero level
        print(f"\nDetermining zero-level for experiment {exp} ...")
        for file in path.glob(f"{exp}*_prob.npy"):

            # get file name and frame for printing
            stack_name = f"{path.parent}{os.sep}raw/{file.name.split('_prob.npy')[0]}.tiff"
            frame = int(stack_name.split('_')[-1].split('.tiff')[0])
            print(f"Frame {frame}")

            # load stacks and ML-predictions
            stack = commonStackReader(stack_name)
            MlM_probabilities = np.load(file)
            cell_prob = MlM_probabilities[:,:,:,1]

            z0, n_z, dn_dz = estimate_cell_bottom(stack)
            n_z_list.append(n_z)
            dn_dz_list.append(dn_dz)
                
            # compute array for determination of MlM threshold
            sum_above = np.zeros_like(thresholds)

            for i in range(len(thresholds)):
                mask = (cell_prob > thresholds[i])
                sum_above[i] = np.sum(mask[z0:])

            # compute threshold
            threshold = determine_threshold(thresholds, sum_above)
            log.write(f"{frame}, z0: {z0}, p_thres: {threshold}\n")

            # apply threshold and split 
            cell_pred  = (cell_prob > threshold)

            # filter mask
            cell_mask = median(cell_pred, kernel)

            # save mask
            out_mask = f"{out_dir}{os.sep}{file.name.split('_prob.npy')[0]}_mask.tiff"
            tifffile.imwrite(out_mask, np.array(cell_mask, dtype=np.uint8), bigtiff=True)



        # save as pickle
        out_dict = {'ri_z_list':        n_z_list,
                    'dri_dz_list':      dn_dz_list,
                    'thresholds':       thresholds,
                    'sum_above':        sum_above,
                    'cell_prob_shape':  cell_prob.shape}
        
        with open(f"{mhds_dir}{os.sep}/lists_for_plotting.pkl", 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)