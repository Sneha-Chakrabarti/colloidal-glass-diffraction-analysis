"""
Burgers vector and quadrupolar strain from local distortion tensors.

Implements the continuous Burgers vector and quadrupolar field as geometric
indicators of local plasticity in glasses, following:

    Liu et al., Acta Cryst. A 82, 4-17 (2026)
    Baggioli et al., Phys. Rev. Lett. 127, 015501 (2021)

The distortion tensor e_ij is measured from nearest-neighbor configurations
(before and after deformation) using a least-squares fit. The Burgers vector
is computed as a closed contour integral of the distortion tensor components.
The quadrupolar field captures the deviatoric (volume-conserving) shear component.

Key equations:
    Burgers vector (Eq. 1, Acta Cryst 2026):
        b_i = -oint du_i = -oint (de_ij/dx_j) dx_j

    Distortion tensor from neighbors (Eq. 13):
        R_alpha = e_ij * R0 * n_hat_alpha

    Quadrupolar field (Eq. 2, 12):
        epsilon_ij = sym(e_ij);  Q_ij = epsilon_ij - (1/2) Tr(epsilon) * I
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Distortion tensor fitting from nearest-neighbor configurations

def fit_distortion_tensor(positions_center, positions_neighbors,
                           box_size, R0=None):
    """
    Fit the local distortion tensor e_ij from nearest-neighbor configuration.

    Solves the least squares problem:
        R_alpha = e_ij * R0 * n_hat_{alpha,i}

    where R_alpha is the vector from central particle to neighbor alpha,
    n_hat is the unit vector in the reference configuration, and R0 is
    the reference (mean) neighbor distance.

    This is Eq. 13 of Acta Cryst 2026.

    Parameters
    ----------
    positions_center    : (2,) array — central particle position
    positions_neighbors : (M, 2) array — neighbor positions
    box_size            : float
    R0                  : float or None — reference nn distance

    Returns
    -------
    e_ij : (2, 2) array — distortion tensor
    residual : float    — fit residual
    """
    M = len(positions_neighbors)
    if M < 3:
        return np.eye(2), np.inf

    # Vectors to neighbors with periodic boundary conditions
    diff = positions_neighbors - positions_center      # (M, 2)
    diff -= box_size * np.round(diff / box_size)

    dists = np.linalg.norm(diff, axis=1)              # (M,)
    if R0 is None:
        R0 = dists.mean()

    # Unit vectors (reference directions)
    valid = dists > 1e-9
    if valid.sum() < 3:
        return np.eye(2), np.inf

    n_hat = diff[valid] / dists[valid, None]           # (M_valid, 2)
    R_ref = R0 * n_hat                                 # (M_valid, 2)

    # Actual vectors
    R_actual = diff[valid]                             # (M_valid, 2)

    # Least squares: R_actual = e_ij @ R_ref^T  (per component)
    # Flatten: for each neighbor alpha, R_actual_i = e_ij * R_ref_j
    # Build design matrix A (M_valid*2, 4) and RHS b (M_valid*2,)
    M_v = valid.sum()
    A = np.zeros((M_v * 2, 4))
    b = np.zeros(M_v * 2)

    for k in range(M_v):
        # Row for x component:  R_actual[k,0] = e00*R_ref[k,0] + e01*R_ref[k,1]
        A[2*k,   0] = R_ref[k, 0]
        A[2*k,   1] = R_ref[k, 1]
        b[2*k]      = R_actual[k, 0]
        # Row for z component:  R_actual[k,1] = e10*R_ref[k,0] + e11*R_ref[k,1]
        A[2*k+1, 2] = R_ref[k, 0]
        A[2*k+1, 3] = R_ref[k, 1]
        b[2*k+1]    = R_actual[k, 1]

    result = np.linalg.lstsq(A, b, rcond=None)
    x = result[0]
    e_ij = np.array([[x[0], x[1]],
                     [x[2], x[3]]])

    residual = float(np.sum((A @ x - b)**2))
    return e_ij, residual


def distortion_field_from_positions(positions, neighbor_list, box_size):
    """
    Compute distortion tensor field e_ij(r) for all particles.

    Parameters
    ----------
    positions     : (N, 2)
    neighbor_list : list of N index arrays
    box_size      : float

    Returns
    -------
    e_field : (N, 2, 2) array — distortion tensor at each particle
    residuals : (N,) array
    """
    N = len(positions)
    e_field   = np.zeros((N, 2, 2))
    residuals = np.zeros(N)

    for i in range(N):
        nbr_idx = neighbor_list[i]
        if len(nbr_idx) == 0:
            e_field[i] = np.eye(2)
            residuals[i] = 0.0
            continue

        e_ij, res = fit_distortion_tensor(
            positions[i],
            positions[nbr_idx],
            box_size,
        )
        e_field[i]   = e_ij
        residuals[i] = res

    return e_field, residuals


def delta_distortion_field(e_before, e_after):
    """
    Compute the deformation-induced distortion: e_d = e_after - e_before.

    This gives the distortion due to the applied deformation increment,
    removing the quenched-in distortions of the glass.

    Parameters
    ----------
    e_before : (N, 2, 2)
    e_after  : (N, 2, 2)

    Returns
    -------
    e_delta : (N, 2, 2)
    """
    return e_after - e_before


# Interpolation to regular grid


def interpolate_to_grid(positions, values, box_size, grid_res=50):
    """
    Interpolate particle-level scalar or vector field to a regular grid.

    Parameters
    ----------
    positions : (N, 2)
    values    : (N,) or (N, K) array
    box_size  : float
    grid_res  : int — grid points per side

    Returns
    -------
    grid_x, grid_z : (grid_res, grid_res) meshgrids
    grid_vals      : (grid_res, grid_res) or (grid_res, grid_res, K) array
    """
    xi = np.linspace(0, box_size, grid_res)
    zi = np.linspace(0, box_size, grid_res)
    grid_x, grid_z = np.meshgrid(xi, zi)

    points = positions  # (N, 2)

    if values.ndim == 1:
        grid_vals = griddata(points, values, (grid_x, grid_z),
                             method='linear', fill_value=float(np.nanmean(values)))
    else:
        K = values.shape[1]
        grid_vals = np.zeros((grid_res, grid_res, K))
        for k in range(K):
            grid_vals[:, :, k] = griddata(
                points, values[:, k], (grid_x, grid_z),
                method='linear', fill_value=float(np.nanmean(values[:, k]))
            )

    return grid_x, grid_z, grid_vals


def distortion_tensor_to_grid(positions, e_field, box_size, grid_res=50):
    """
    Interpolate the (N, 2, 2) distortion tensor field to a regular grid.

    Returns
    -------
    grid_x, grid_z : (G, G) meshgrids
    e_grid : (G, G, 2, 2) array
    """
    G = grid_res
    e_flat = e_field.reshape(-1, 4)   # (N, 4)
    grid_x, grid_z, e_grid_flat = interpolate_to_grid(
        positions, e_flat, box_size, grid_res=G
    )
    e_grid = e_grid_flat.reshape(G, G, 2, 2)
    return grid_x, grid_z, e_grid


# Gradient computation (for Burgers vector)


def compute_gradients(e_grid, box_size, grid_res=50):
    """
    Compute spatial gradients of the distortion tensor components.

    de_ij / dx and de_ij / dz using finite differences on the grid.

    Parameters
    ----------
    e_grid  : (G, G, 2, 2) array — distortion tensor on regular grid
    box_size : float
    grid_res : int

    Returns
    -------
    de_dx : (G, G, 2, 2) — de_ij/dx
    de_dz : (G, G, 2, 2) — de_ij/dz
    """
    dx = box_size / grid_res
    dz = box_size / grid_res

    de_dx = np.gradient(e_grid, dx, axis=1)   # gradient along x (axis=1)
    de_dz = np.gradient(e_grid, dz, axis=0)   # gradient along z (axis=0)

    return de_dx, de_dz


# Burgers vector


def compute_burgers_vector(e_grid, box_size, grid_res=50,
                            contour_size=7, smooth_sigma=1.0):
    """
    Compute the local Burgers vector field from the distortion tensor grid.

    The Burgers vector is calculated as a closed square contour integral
    (Eq. 8, Acta Cryst 2026; Baggioli et al. 2021):

        b_i = -oint e_ij dx_j

    The contour is a square of side 2L centered at each grid point.

    Parameters
    ----------
    e_grid      : (G, G, 2, 2) distortion tensor grid
    box_size    : float
    grid_res    : int
    contour_size : int — contour square half-width in grid pixels (paper uses 7)
    smooth_sigma : float — Gaussian smoothing of e_grid before differentiation

    Returns
    -------
    burgers : (G, G, 2) array — Burgers vector (bx, bz) at each grid point
    b_mag   : (G, G) array   — |b| magnitude
    """
    G = grid_res
    dx = box_size / G    # grid spacing

    # Smooth distortion field to reduce noise
    e_smooth = np.zeros_like(e_grid)
    for i in range(2):
        for j in range(2):
            e_smooth[:, :, i, j] = gaussian_filter(e_grid[:, :, i, j],
                                                     sigma=smooth_sigma)

    burgers = np.zeros((G, G, 2))
    L = contour_size // 2

    for zi in range(G):
        for xi in range(G):
            b = np.zeros(2)

            # Four line segments of the square contour (anticlockwise):
            # A: bottom  (xi-L, zi-L) -> (xi+L, zi-L), dz=0, dx varies
            # B: right   (xi+L, zi-L) -> (xi+L, zi+L), dx=0, dz varies
            # C: top     (xi+L, zi+L) -> (xi-L, zi+L), dz=0, dx varies (reversed)
            # D: left    (xi-L, zi+L) -> (xi-L, zi-L), dx=0, dz varies (reversed)

            for seg in range(4):
                if seg == 0:   # A: bottom, z=zi-L, x from xi-L to xi+L
                    z_seg = max(0, min(G-1, zi - L))
                    x_range = range(max(0, xi-L), min(G, xi+L))
                    for x in x_range:
                        e = e_smooth[z_seg, x]
                        b[0] -= e[0, 0] * dx    # e_xx * dx
                        b[1] -= e[1, 0] * dx    # e_zx * dx

                elif seg == 1: # B: right, x=xi+L, z from zi-L to zi+L
                    x_seg = max(0, min(G-1, xi + L))
                    z_range = range(max(0, zi-L), min(G, zi+L))
                    for z in z_range:
                        e = e_smooth[z, x_seg]
                        b[0] -= e[0, 1] * dx    # e_xz * dz
                        b[1] -= e[1, 1] * dx    # e_zz * dz

                elif seg == 2: # C: top, z=zi+L, x from xi+L to xi-L
                    z_seg = max(0, min(G-1, zi + L))
                    x_range = range(min(G-1, xi+L), max(-1, xi-L-1), -1)
                    for x in x_range:
                        e = e_smooth[z_seg, x]
                        b[0] += e[0, 0] * dx
                        b[1] += e[1, 0] * dx

                else:          # D: left, x=xi-L, z from zi+L to zi-L
                    x_seg = max(0, min(G-1, xi - L))
                    z_range = range(min(G-1, zi+L), max(-1, zi-L-1), -1)
                    for z in z_range:
                        e = e_smooth[z, x_seg]
                        b[0] += e[0, 1] * dx
                        b[1] += e[1, 1] * dx

            burgers[zi, xi] = b

    b_mag = np.sqrt(burgers[:, :, 0]**2 + burgers[:, :, 1]**2)
    return burgers, b_mag


def compute_burgers_fast(e_grid, box_size, grid_res=50,
                          contour_size=7, smooth_sigma=1.0):
    """
    Fast Burgers vector via Stokes theorem: b = curl of distortion tensor.

    By Stokes' theorem, the Burgers integral equals the surface integral
    of the curl of e_ij. This avoids the nested loop.

        b_x = d(e_xz)/dx - d(e_xx)/dz  (from x-component curl)
        b_z = d(e_zz)/dx - d(e_zx)/dz  (from z-component curl)

    Parameters match compute_burgers_vector().
    """
    G = grid_res
    dx = box_size / G

    e_smooth = np.zeros_like(e_grid)
    for i in range(2):
        for j in range(2):
            e_smooth[:, :, i, j] = gaussian_filter(e_grid[:, :, i, j],
                                                     sigma=smooth_sigma)

    # Gradients: de_ij/dx and de_ij/dz
    # np.gradient: axis=1 is x, axis=0 is z
    de_dx = np.gradient(e_smooth, dx, axis=1)   # (G, G, 2, 2)
    de_dz = np.gradient(e_smooth, dx, axis=0)

    # Curl components (antisymmetric part of de_ij/dx_k)
    # b_x: integral of (de_xz/dx - de_xx/dz) * area
    # b_z: integral of (de_zz/dx - de_zx/dz) * area
    curl_x = de_dx[:, :, 0, 1] - de_dz[:, :, 0, 0]   # (G, G)
    curl_z = de_dx[:, :, 1, 1] - de_dz[:, :, 1, 0]

    # Multiply by contour area (2L * 2L * dx^2)
    L = contour_size // 2
    area = (2 * L * dx) ** 2

    burgers = np.stack([curl_x * area, curl_z * area], axis=-1)   # (G, G, 2)
    b_mag   = np.sqrt(burgers[:, :, 0]**2 + burgers[:, :, 1]**2)

    return burgers, b_mag

# Quadrupolar strain field


def compute_quadrupole(e_grid):
    """
    Compute the quadrupolar strain field from the distortion tensor grid.

    The quadrupolar field is the traceless symmetric part of the strain:
        epsilon_ij = (e_ij + e_ji) / 2       (symmetric strain)
        Q_ij = epsilon_ij - (1/2) Tr(epsilon) * I    (traceless = pure shear)

    Magnitude and direction (Eq. 10-12, Acta Cryst 2026):
        Q_mag = sqrt(Q_xx^2 + Q_xz^2)
        theta = (1/2) arctan(Q_xz / Q_xx)

    Parameters
    ----------
    e_grid : (G, G, 2, 2)

    Returns
    -------
    Q_mag   : (G, G)
    Q_dir   : (G, G)   — direction angle theta
    Q_field : (G, G, 2, 2)
    """
    # Symmetric strain
    eps = (e_grid + e_grid.transpose(0, 1, 3, 2)) / 2.0   # (G, G, 2, 2)

    # Trace
    tr = eps[:, :, 0, 0] + eps[:, :, 1, 1]   # (G, G)

    # Traceless part
    Q = eps.copy()
    Q[:, :, 0, 0] -= tr / 2
    Q[:, :, 1, 1] -= tr / 2

    Q11 = Q[:, :, 0, 0]
    Q12 = Q[:, :, 0, 1]

    Q_mag = np.sqrt(Q11**2 + Q12**2)
    Q_dir = 0.5 * np.arctan2(Q12, Q11)

    return Q_mag, Q_dir, Q


# Summary statistics (Fig 4 analog)

def parameter_vs_centrosymmetry_quartiles(F_IS, param, n_quartiles=4):
    """
    Compute mean and std of a parameter for each quartile of F_IS.

    Reproduces the analysis in Acta Cryst 2026 Fig. 4:
    average Burgers/quadrupole magnitude vs quartile of centrosymmetry.

    Parameters
    ----------
    F_IS  : (N,) array — centrosymmetry values
    param : (N,) array — parameter (e.g. Burgers magnitude)
    n_quartiles : int

    Returns
    -------
    quartile_centers : (n_quartiles,) — mean F_IS in each quartile
    param_means      : (n_quartiles,) — mean param in each quartile
    param_stds       : (n_quartiles,) — std param in each quartile
    """
    percentiles = np.linspace(0, 100, n_quartiles + 1)
    boundaries  = np.percentile(F_IS, percentiles)

    centers  = np.zeros(n_quartiles)
    means    = np.zeros(n_quartiles)
    stds     = np.zeros(n_quartiles)

    for q in range(n_quartiles):
        mask = (F_IS >= boundaries[q]) & (F_IS < boundaries[q + 1])
        if q == n_quartiles - 1:
            mask = (F_IS >= boundaries[q]) & (F_IS <= boundaries[q + 1])
        if mask.sum() == 0:
            continue
        centers[q] = F_IS[mask].mean()
        means[q]   = param[mask].mean()
        stds[q]    = param[mask].std()

    return centers, means, stds
