import numpy as np
import torch

def normalizeY(posxyz):
    '''
    Normalize on Y axis by scaling with the highest position recorded
    --------------
    :param
    posxyz: Matrix with every X,Y and Z positions for every trajectory
    --------------
    :return
    posxyz: Same matrix as provided with Y positions normalized
    '''

    # Find maximum y position
    onlyposy = torch.flatten(posxyz[:, 1, :])
    maxy = onlyposy[torch.argmax(onlyposy)].item()

    # Divide all y values by maxy to normalize height
    posxyz[:, 1, :] = posxyz[:, 1, :] / maxy

    return posxyz

def normalizeZ(posxyz):
    '''
    Normalize on Z axis by computing distance with the center of mass
    --------------
    :param
    posxyz: Matrix with every X,Y and Z positions for every trajectory
    --------------
    :return
    posxyz: Same matrix as provided with Z positions normalized
    '''

    # Find center of mass on z axis for each frames (mean of all z pos for one frame)
    onlyposz = posxyz[:,2,:]
    com = torch.mean(onlyposz, dim=0)

    # Normalize all z positions by center of mass (difference between com and point)
    posxyz[:,2,:] = onlyposz - com

    return posxyz

def computeAV(posxyz):
    '''
    Add Acceleration and velocity to our trajectories.
    --------------
    :param
    posxyz: Matrix with every X,Y and Z positions for every trajectory
    --------------
    :return
    The positions matrix concatenated with the acceleration and velocity
    '''

    xyz_v = torch.zeros(posxyz.shape, dtype=posxyz.dtype)
    xyz_v[1:,:,:] = posxyz[1:,:,:] - posxyz[0:posxyz.shape[0]-1,:,:]
    xyz_v_norm = xyz_v.norm(dim=1)
    if xyz_v_norm.shape[0] > 1:
        xyz_v_norm[0,:] = xyz_v_norm[1,:]
    xyz_a = torch.zeros(posxyz.shape,dtype=posxyz.dtype)
    xyz_a[1:,:,:] = xyz_v[1:xyz_v.shape[0],:,:] - xyz_v[0:xyz_v.shape[0]-1,:,:]
    xyz_a_norm = xyz_a.norm(dim=1)
    if xyz_a_norm.shape[0] > 2:
        xyz_a_norm[1,:] = xyz_a_norm[2,:]
        xyz_a_norm[0,:] = xyz_a_norm[2,:]

    xyz_v_norm = xyz_v_norm[:, None, :]
    xyz_a_norm = xyz_a_norm[:, None, :]

    return torch.cat((posxyz,xyz_v_norm, xyz_a_norm),1)

def linInterpolation(posxyz):
    '''
    Fills NaN values with linear interpolation
    --------------
    :param
    posxyz: Matrix with every X,Y and Z positions for every trajectory
    --------------
    :return
    posxyz: Positions matrix without the NaN values treated
    '''

    for i in range(len(posxyz)):
        nans, x= np.isnan(posxyz[i]), lambda z: z.nonzero()[0]
        posxyz[i][nans]= np.interp(x(nans), x(~nans), posxyz[i][~nans])
    return posxyz

def cleanDirection(posxyz):
    '''
    Delete all entries where subject is not going toward positive X.
    Warning: This as caused very bad results!!
    --------------
    :param
    posxyz: Matrix with every X,Y and Z positions for every trajectory
    --------------
    :return
    out: Only the trajectories facing positive X
    '''

    posxyz = posxyz.numpy()
    datasToKeep = []
    for i in range(posxyz.shape[0]):
        if posxyz[i,0,0].item() < 0:
            datasToKeep.append(posxyz[i,:,:])
    out = torch.from_numpy(np.array(datasToKeep))
    return out