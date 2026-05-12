'''
Functions for 
'''

import numpy as np
import scipy as sc



def split_tiles(stack, tile_size=850, full_tile_size=958, Ntiles=4):
    '''
    Split a 3D stack into tiles with edge removed.

    Parameters:
    - stack: 3D numpy array of shape (Nz, Nxy, Nxy), where:
        - Nz: Number of slices in the z-dimension.
        - Nxy: Size of the xy dimensions.
    - tile_size: The size of the tile to extract (default is 850).
    - full_tile_size: The size of the full tile before edge removal (default is 958).
    - Ntiles: The number of tiles to create along each axis (default is 4).

    Returns:
    - tiles: A 5D numpy array of shape (Ntiles, Ntiles, Nz, tile_size, tile_size),
             containing the extracted tiles.
    '''

    # Get dimensions of the input stack
    Nz, Nxy, _ = np.shape(stack)  # Extract the number of z slices and size of xy dimension
    dxy = int(tile_size / 2)      # Calculate half the tile size for centered indexing

    # Define the edge indices for extracting tiles ( centering the extraction within bounds )
    xy_edge = np.array([0, Nxy / 2 - full_tile_size, Nxy / 2, Nxy / 2 + full_tile_size, Nxy], dtype=int)
    
    # Initialize an array to hold the extracted tiles
    tiles = np.zeros([Ntiles, Ntiles, Nz, tile_size, tile_size])

    # Loop over each tile's position
    for ix in range(Ntiles):
        for iy in range(Ntiles):
            # Extract a tile from the stack using the defined edge indices
            tile = stack[:, xy_edge[iy]:xy_edge[1 + iy], xy_edge[ix]:xy_edge[1 + ix]]
            
            # Get the current tile's dimensions
            dims = np.shape(tile)
            xmid = int(dims[2] / 2)  # Calculate the mid-point for x dimension
            ymid = int(dims[1] / 2)  # Calculate the mid-point for y dimension

            # Center the tile by removing the edges using calculated half-widths (dxy)
            tiles[iy, ix] = tile[:, ymid - dxy:ymid + dxy, xmid - dxy:xmid + dxy]

    return tiles  # Return the array of tiles



def tile_z0_planes(z0_tile, weight_tile, Nxy=3648, full_tile_size=958, Ntiles=4):
    '''
    Combine fitted z0 planes as tiles. To be used as bottom of full image.
    '''

    # Define the edge indices for extracting tiles ( centering the extraction within bounds )
    xy_edge = np.array([0, Nxy / 2 - full_tile_size, Nxy / 2, Nxy / 2 + full_tile_size, Nxy], dtype=int)
    
    # Calculate the dimensions of the full stack
    full_image = np.zeros([Nxy, Nxy])  # Initialize the full stack array
    mid = int(full_tile_size / 2)

    # Loop through each tile to combine them into the full stack
    for ix in range(Ntiles):
        for iy in range(Ntiles):

            # Get the current tile's dimensions
            dx = int((xy_edge[ix+1] - xy_edge[ix]) / 2)
            dy = int((xy_edge[iy+1] - xy_edge[iy]) / 2)

            # Center the tile by removing the edges using calculated half-widths (dxy)
            z0_plane = z0_tile[mid-dy:mid+dy, mid-dx:mid+dx]
            w = weight_tile[iy, ix]
            full_image[xy_edge[iy]:xy_edge[iy+1], xy_edge[ix]:xy_edge[ix+1]] = z0_plane * w

    return full_image



def estimate_cell_bottom_reference(mask, f_crit=0.5):
    dims = np.shape(mask)

    # compute fraction of voxels that are cell along z-axis
    f_cells = np.sum(mask, axis=(1,2)) / (dims[1] * dims[2])
    zslice  = np.arange(len(f_cells))

    # use first z-slice above f_crit as reference
    zslice_mask = zslice[f_cells > f_crit]
    z0 = zslice_mask[np.argmin(zslice_mask)]

    return z0



def update_cell_mask(mask, z0, z0_plane):

    new_mask = np.copy(mask).astype(float)
    new_mask[z0] = 1
    new_mask[:z0] = 0
    new_mask[z0-1] = z0_plane
    new_mask[z0-2] = (z0_plane-1) * (z0_plane > 1)

    return new_mask



# Fit plane to cell bottom
def plane(params, x, y):
    a, b, c = params
    return a*x + b*y + c

def residual_plane(params, X, Y, Z):
    return Z - plane(params, X, Y)


def fit_plane(Z0, full_dims):
    '''
    Fit linear plane to z0 data points and returning plane for full tile
    '''

    dims = np.shape(Z0)
    Y, X = np.meshgrid(np.arange(dims[0]), 
                       np.arange(dims[1]))

    # tile without edges
    X = X.flatten()
    Y = Y.flatten()
    Z = Z0.flatten()

    flat_params = np.array([0,0, np.mean(Z0)])
    result = sc.optimize.least_squares(residual_plane, flat_params, args=(X, Y, Z))

    # tile with edges
    dim_diff = int(full_dims[0] - dims[0]) / 2
    Y, X = np.meshgrid(np.arange(full_dims[0]) - dim_diff, 
                       np.arange(full_dims[1]) - dim_diff)

    X = X.flatten()
    Y = Y.flatten()

    z_plane = plane(result.x, X, Y)

    return z_plane.reshape(full_dims[0], full_dims[1])
