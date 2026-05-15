'''
Functions for transforming Tomocube data to 'Holomonitor data', i.e. 2D tiffs of heights and mean refractive indices
'''

import numpy as np
import scipy as sc
from skimage.morphology import disk


def estimate_cell_bottom(stack):
    '''
    Estimates first z-slice with cells.
    Assumes it is where derivative of refractive index is max.
    dn_dz = np.diff(np.mean(n, axis=(1,2))), i.e. the derivative along z of the mean refractive index of each stack
    '''

    n_z   = np.mean(stack, axis=(1,2))
    dn_dz = np.diff(n_z) + 1
    z0    = np.argmax(dn_dz / n_z[:-1])

    return z0, n_z, dn_dz



def determine_threshold(thresholds, sum_mask):
    '''
    Determine threshold to be used to distinguish cell from media.
    Uses threshold that minimizes magnitude of derivative of cell mask.
    '''

    centered_thresholds = (thresholds[1:] + thresholds[:-1]) / 2
    dsum_mask = np.diff(sum_mask)
    idx = np.argmin(abs(dsum_mask))

    return centered_thresholds[idx]



def generate_kernel(r_min, r_max):
    '''
    Creates 3D kernel that is used to median filter the cell mask.
    '''
    r_mid = r_min + int((r_max-r_min) / 2)
    p_min = r_max - r_min
    p_mid = r_max - r_mid

    kernel = np.array([np.pad(disk(r_min), pad_width=p_min),
                       np.pad(disk(r_mid), pad_width=p_mid),
                       disk(r_max),
                       np.pad(disk(r_mid), pad_width=p_mid),
                       np.pad(disk(r_min), pad_width=p_min)])
    
    return kernel
