import numpy as np

def bipolar_Avec(coordinates:np.ndarray, q_para:float, L_para:float, d_para:float):

    """
    Calculate the bipolar vector potential Avec
    """
    x, y, z = coordinates[0], coordinates[1], coordinates[2]
    Avec = np.zeros((3, *x.shape))
    cos1 = (x-L_para) / np.sqrt(y**2+(z+d_para)**2+(x-L_para)**2)
    cos2 = (L_para+x) / np.sqrt(y**2+(z+d_para)**2+(x+L_para)**2)
    Aphi  = cos1 * q_para - cos2 * q_para

    Avec[0] = 0.0
    Avec[1] = -Aphi * (z+d_para)/(y**2+(z+d_para)**2)
    Avec[2] = Aphi * y/(y**2+(z+d_para)**2)

    return Avec

def bipolar_Bvec(coordinates:np.ndarray, q_para:float, L_para:float, d_para:float):

    """
    Calculate the bipolar magnetic field Bvec
    """
    x, y, z = coordinates[0], coordinates[1], coordinates[2]
    Bvec = np.zeros((3, *x.shape))
    tmp = np.sqrt(y**2+(z+d_para)**2+(x+L_para)**2)**3
    Bvec[0] = (x+L_para)/tmp
    Bvec[1] = y/tmp
    Bvec[2] = (z+d_para)/tmp
    tmp = np.sqrt(y**2+(z+d_para)**2+(x-L_para)**2)**3
    Bvec[0] = Bvec[0] - (x-L_para)/tmp
    Bvec[1] = Bvec[1] - y/tmp
    Bvec[2] = Bvec[2] - (z+d_para)/tmp

    return Bvec


def rbsl_Avec(coordinates:np.ndarray, x_axis:np.ndarray, a:float, F_flx:float, positive_helicity:bool):
    """
    Calculate the vector potential and magnetic field for a flux rope in Cartesian coordinates.

    Parameters:
    coordinates : np.ndarray    # AIx = np.zeros_like(coordinates)
    # AFx = np.zeros_like(coordinates)
        Coordinates where to calculate the fields, shape (3,N)
    x_axis : np.ndarray 
        Coordinates defining the flux rope axis, shape (3,M) (the closed loop)
    a : float
        Cross-sectional radius of the flux rope.
    F_flx : float
        Net magnetic flux along the flux rope axis.
    positive_helicity : bool
        Indicates if the helicity is positive.

    Returns:
    Atotal : np.ndarray
        Vector potential.
    """

    naxis = x_axis.shape[1]

    # Calculate I_cur
    I_cur = 1.0 * F_flx * 5.0 * np.sqrt(2.0) / 3.0 / a
    re_pi = 1.0 / np.pi

    r_vec = (coordinates[:, :, None] - x_axis[:, None, :]) / a # r_vec.shape = (3, N, M)
    r_mag = np.linalg.norm(r_vec, axis=0) # r_mag.shape = (N, M)

    Rpl = np.zeros_like(x_axis) # Rpl.shape = (3, M)
    Rpl[:, 1:-1] = 0.5 * (x_axis[:, 2:] - x_axis[:, :-2])
    Rpl[:, 0] = 0.5 * (x_axis[:, 1] - x_axis[:, -1])
    Rpl[:, -1] = 0.5 * (x_axis[:, 0] - x_axis[:, -2])
    Rcr = np.cross(Rpl[:, None, :], r_vec, axis=0) # Rcr.shape = (3, N, M)

    mask = r_mag <= 1.0
    r_mag1 = r_mag[mask]
    r_mag2 = r_mag[~mask]

    KIr = np.zeros(r_mag.shape)
    KFr = np.zeros(r_mag.shape) # KIr.shape = (N, M)

    # KIr = np.where(mask, 2.0*re_pi*((np.arcsin(r_mag)) / (r_mag) + (5.0 - 2.0 * r_mag**2) / 3.0 * np.sqrt(1 - r_mag**2)), 1.0/r_mag)
    KIr[mask] = 2.0*re_pi*((np.arcsin(r_mag1)) / (r_mag1) + (5.0 - 2.0 * r_mag1**2) / 3.0 * np.sqrt(1 - r_mag1**2))
    KIr[~mask] = 1.0/r_mag2

    # KFr = np.where(mask, 2.0*re_pi/r_mag**2*((np.arcsin(r_mag))/(r_mag)-np.sqrt(1-r_mag**2) + 
    #                 2.0*re_pi*np.sqrt(1-r_mag**2)+(5.0-2.0*(r_mag**2))*0.5/np.sqrt(6.0)*(1.0- 
    #                 2.0*re_pi*np.arcsin((1.0+2.0*r_mag**2)/(5.0-2.0*r_mag**2)))), KIr**3)
    KFr[mask] = 2.0*re_pi/r_mag1**2*((np.arcsin(r_mag1))/(r_mag1)-np.sqrt(1-r_mag1**2) + 
                    2.0*re_pi*np.sqrt(1-r_mag1**2)+(5.0-2.0*(r_mag1**2))*0.5/np.sqrt(6.0)*(1.0- 
                    2.0*re_pi*np.arcsin((1.0+2.0*r_mag1**2)/(5.0-2.0*r_mag1**2))))
    KFr[~mask] = KIr[~mask]**3


    AIx = np.sum(I_cur * 0.25 * re_pi * KIr[None, :, :] * Rpl[:, None, :] / a, axis=2)
    AFx = np.sum(F_flx * 0.25 * re_pi * KFr[None, :, :] * Rcr / a**2, axis=2)

    return AIx + AFx

def curl_slab(vec:np.ndarray, dxyz):
    """
    Calculate the curl of a vector field in Cartesian coordinates.
    
    Parameters:
        vec: np.ndarray with shape (3, nx, ny, nz) representing vector field components
        dxyz: Iterable with the grid spacing, shape (3,)

    Returns:
        curl: np.ndarray with shape (3, nx-2, ny-2, nz-2) representing the curl of the vector field
        The curl is calculated at the cell centers.
    """
    curl = np.zeros_like(vec)
    
    # Calculate partial derivatives for each component
    # dw/dy - dv/dz
    curl[0, ...] = np.gradient(vec[2, ...], dxyz[1], axis=1) - \
                   np.gradient(vec[1, ...], dxyz[2], axis=2)
    
    # du/dz - dw/dx  
    curl[1, ...] = np.gradient(vec[0, ...], dxyz[2], axis=2) - \
                   np.gradient(vec[2, ...], dxyz[0], axis=0)
    
    # dv/dx - du/dy
    curl[2, ...] = np.gradient(vec[1, ...], dxyz[0], axis=0) - \
                   np.gradient(vec[0, ...], dxyz[1], axis=1)

    return curl[:, 1:-1, 1:-1, 1:-1]

def TDm_slab(xmin, xmax, domain_nx, r0:float, a0:float, ispositive:bool, 
             naxis:int, q0:float, L0:float, d0:float, knonb:float=1.0):

    """
    Calculate the TDm model from rbsl for a slab domain.

    """

    # Ensure inputs are numpy arrays
    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax) 
    domain_nx = np.asarray(domain_nx)

    # Verify shapes
    assert xmin.shape == (3,), f"xmin must be array of shape (3,), got {xmin.shape}"
    assert xmax.shape == (3,), f"xmax must be array of shape (3,), got {xmax.shape}"
    assert domain_nx.shape == (3,), f"domain_nx must be array of shape (3,), got {domain_nx.shape}"

    # Define the grid
    xrange = np.linspace(xmin[0], xmax[0], domain_nx[0]+1)[:-1] + 0.5 * (xmax[0] - xmin[0]) / domain_nx[0]
    yrange = np.linspace(xmin[1], xmax[1], domain_nx[1]+1)[:-1] + 0.5 * (xmax[1] - xmin[1]) / domain_nx[1]
    zrange = np.linspace(xmin[2], xmax[2], domain_nx[2]+1)[:-1] + 0.5 * (xmax[2] - xmin[2]) / domain_nx[2]
    x_mesh, y_mesh, z_mesh = np.meshgrid(xrange, yrange, zrange, indexing='ij')
    coordinates = np.stack([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()])

    dxyz = (xmax-xmin) / domain_nx

    # Define the grid for the extended domain for derivatives
    xrange1 = np.concatenate([xrange[0:1]-dxyz[0], xrange, xrange[-1:]+dxyz[0]])
    yrange1 = np.concatenate([yrange[0:1]-dxyz[1], yrange, yrange[-1:]+dxyz[1]])
    zrange1 = np.concatenate([zrange[0:1]-dxyz[2], zrange, zrange[-1:]+dxyz[2]])
    x_mesh1, y_mesh1, z_mesh1 = np.meshgrid(xrange1, yrange1, zrange1, indexing='ij')
    coordinates1 = np.stack([x_mesh1.flatten(), y_mesh1.flatten(), z_mesh1.flatten()])

    h0 = r0 - d0
    x0 = (xmin[0] + xmax[0]) / 2.0
    y0 = (xmin[1] + xmax[1]) / 2.0
    z0 = -d0

    Bperp = q0*(L0/np.sqrt(r0**2+L0**2)**3+L0/np.sqrt(r0**2-L0**2)**3)
    # theta = np.arctan(np.sqrt(r0**2-d0**2)/d0) no need for this by a closed loop
    I_cur = (r0*Bperp/(np.log(8*r0/a0)-1))*knonb
    F_flx = 4.0*np.pi*3*0.2/np.sqrt(2.0)*I_cur*a0

    x_axis = np.zeros((3, naxis))
    for i in range(naxis):
        x_axis[0, i] = x0
        x_axis[1, i] = y0 + r0*np.sin(2*np.pi*i/naxis)*np.sign(F_flx)
        x_axis[2, i] = z0 + r0*np.cos(2*np.pi*i/naxis)

    Avec_rbsl = rbsl_Avec(coordinates1, x_axis, a0, F_flx, ispositive).reshape(3, *(domain_nx+2))
    Bvec_rbsl = curl_slab(Avec_rbsl, dxyz)
    Bvec_bipolar = bipolar_Bvec(coordinates, q0, L0, d0).reshape(3, *domain_nx)

    return (Bvec_rbsl + Bvec_bipolar)

def dipolez_Avec(coordinates:np.ndarray, mz:float, posi:np.ndarray):
    """
    Calculate the vector potential of a dipole in the z-direction.

    input:
    coordinates: np.ndarray, shape (3, ...)
    posi: np.ndarray, shape (3,)
    """

    x, y, z = coordinates[0], coordinates[1], coordinates[2]
    xi, yi, zi = posi[0], posi[1], posi[2]
    r = np.sqrt((x-xi)**2 + (y-yi)**2 + (z-zi)**2)

    Avec = np.zeros((3, *x.shape))
    Avec[0] = -mz * (y-yi) / r**3
    Avec[1] = mz * (x-xi) / r**3
    Avec[2] = 0.0

    return Avec

def dipole_Bvec(coordinates:np.ndarray, m:np.ndarray, posi:np.ndarray):
    """
    Calculate the magnetic field of a dipole.
    """
    x, y, z = coordinates[0], coordinates[1], coordinates[2]
    xi, yi, zi = posi[0], posi[1], posi[2]
    r = np.sqrt((x-xi)**2 + (y-yi)**2 + (z-zi)**2)

    Bvec = np.zeros((3, *x.shape))

    # Calculate dot product components explicitly
    dot_product = (m[0]*(x-xi) + m[1]*(y-yi) + m[2]*(z-zi))

    Bvec[0] = 3 * (x-xi) * dot_product / r**5 - m[0] / r**3
    Bvec[1] = 3 * (y-yi) * dot_product / r**5 - m[1] / r**3
    Bvec[2] = 3 * (z-zi) * dot_product / r**5 - m[2] / r**3

    return Bvec


def monopole_Bvec(coordinates:np.ndarray, q:float, posi:np.ndarray):
    """
    Calculate the magnetic field of a monopole.
    """
    x, y, z = coordinates[0], coordinates[1], coordinates[2]
    xi, yi, zi = posi[0], posi[1], posi[2]
    r = np.sqrt((x-xi)**2 + (y-yi)**2 + (z-zi)**2)

    Bvec = np.zeros((3, *x.shape))
    Bvec[0] = q*(x-xi) / r**3
    Bvec[1] = q*(y-yi) / r**3
    Bvec[2] = q*(z-zi) / r**3

    return Bvec

def fan_Avec(coordinates:np.ndarray, poses:np.ndarray, mz):
    """
    Calculate the vector potential of a fan.

    coordinates: np.ndarray, shape (3, ...)
    poses: np.ndarray, shape (M, 3), M set the first dim for memory efficiency in loop
    """

    x, y, z = coordinates[0], coordinates[1], coordinates[2]

    Avec = np.zeros((3, *x.shape))

    for i in range(poses.shape[0]):
        Avec += dipolez_Avec(coordinates, mz[i], poses[i])

    return Avec

def fan_Bvec(coordinates:np.ndarray, poses:np.ndarray, m:np.ndarray):
    """
    Calculate the magnetic field of a fan.

    """
    Bvec = np.zeros(coordinates.shape)
    for i in range(poses.shape[0]):
        Bvec += dipole_Bvec(coordinates, m[i], poses[i])

    return Bvec

def fan_slab(xmin:np.ndarray, xmax:np.ndarray, domain_nx:np.ndarray, poses:np.ndarray, m:np.ndarray):

    # Ensure inputs are numpy arrays
    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax) 
    domain_nx = np.asarray(domain_nx)

    # Verify shapes
    assert xmin.shape == (3,), f"xmin must be array of shape (3,), got {xmin.shape}"
    assert xmax.shape == (3,), f"xmax must be array of shape (3,), got {xmax.shape}"
    assert domain_nx.shape == (3,), f"domain_nx must be array of shape (3,), got {domain_nx.shape}"

    # Define the grid
    xrange = np.linspace(xmin[0], xmax[0], domain_nx[0]+1)[:-1] + 0.5 * (xmax[0] - xmin[0]) / domain_nx[0]
    yrange = np.linspace(xmin[1], xmax[1], domain_nx[1]+1)[:-1] + 0.5 * (xmax[1] - xmin[1]) / domain_nx[1]
    zrange = np.linspace(xmin[2], xmax[2], domain_nx[2]+1)[:-1] + 0.5 * (xmax[2] - xmin[2]) / domain_nx[2]
    x_mesh, y_mesh, z_mesh = np.meshgrid(xrange, yrange, zrange, indexing='ij')
    coordinates = np.stack([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()])

    dxyz = (xmax-xmin) / domain_nx

    # Define the grid for the extended domain for derivatives
    xrange1 = np.concatenate([xrange[0:1]-dxyz[0], xrange, xrange[-1:]+dxyz[0]])
    yrange1 = np.concatenate([yrange[0:1]-dxyz[1], yrange, yrange[-1:]+dxyz[1]])
    zrange1 = np.concatenate([zrange[0:1]-dxyz[2], zrange, zrange[-1:]+dxyz[2]])
    x_mesh1, y_mesh1, z_mesh1 = np.meshgrid(xrange1, yrange1, zrange1, indexing='ij')
    coordinates1 = np.stack([x_mesh1.flatten(), y_mesh1.flatten(), z_mesh1.flatten()])

    fanB = fan_Bvec(coordinates, poses, m).reshape(3, *(domain_nx))

    return fanB
