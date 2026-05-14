'''
Functions for transforming Tomocube data to 'Holomonitor data', i.e. 2D tiffs of heights and mean refractive indices
'''

import numpy as np



def compute_height(cell_pred, method="sum"):
    '''
    Computes cell heights either by summing voxels or taking the difference between min and max.
    Assumes prediction voxels are 0 or 1. Returns height in units of voxels.
    '''
    assert method=="sum" or method=="diff"
    assert np.max(cell_pred) == 1

    if method=="sum":
        h = np.sum(cell_pred, axis=0)

    elif method=="diff":
        _, Z_idx, _ = np.meshgrid(np.arange(0, len(cell_pred[0])),
                                  np.arange(0, len(cell_pred)), 
                                  np.arange(0, len(cell_pred[0,0])))
        Z_idx = Z_idx * cell_pred
        Z_idx[Z_idx==0] = np.nan

        h = np.nanmax(Z_idx, axis=0) - np.nanmin(Z_idx, axis=0) + 1

    return h



def refractive_index_uint16(stack, mask):
    '''
    Transforms float16 to uint16. Used on refractive indices to save as uint16.
    Returns sum and mean
    '''

    # Take average
    ridx_sum  = np.sum(stack * mask, axis=0)
    height    = np.sum(mask, axis=0)
    ridx_avrg = np.copy(ridx_sum)
    ridx_avrg[height > 0]  = ridx_avrg[height > 0] / height[height > 0]

    # remove empty regions
    ridx_sum[height <= 0] = 0
    ridx_avrg[height <= 0] = 0

    return np.array(ridx_avrg, dtype=np.uint16)