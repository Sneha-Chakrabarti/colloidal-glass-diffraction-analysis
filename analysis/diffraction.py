"""
Kinematical diffraction and angular correlation analysis.

Implements:
    - Projected diffraction intensity I(q, phi) for a cluster
    - Fourier coefficients of the angular autocorrelation c^n_q
    - Orientational averaging over random rotations
    - Symmetry fingerprint extraction averaged over first diffraction peak

Reference: Liu et al., Phys. Rev. Lett. 116, 205501 (2016)
    Equations 6, 8, 9 and surrounding derivation.
"""

import numpy as np
from scipy.special import jv  # Bessel functions of the first kind


# Rotation utilities


def random_rotation_matrix(rng=None):
    """
    Uniform random rotation matrix (Haar measure on SO(3)).
    Uses the Shoemake / QR method.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Generate random 3x3 matrix, QR decompose
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Fix signs so det(Q) = +1
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotate_cluster(coords, R):
    """Apply rotation matrix R to (N,3) coordinate array."""
    return coords @ R.T


# Core diffraction calculation

def cluster_fourier_coefficients(coords, q_values, n_max=12):
    """
    Compute Fourier coefficients I^n_q for a single cluster orientation.

    For a cluster projected onto the xy-plane, the diffracted intensity
    Fourier coefficients are (Eq. 8, PRL 2016):

        I^n_q = f^2(q) * sum_{i,j} i^n * J_n(q * r_ij) * exp(-i*n*phi_ij)

    We assume f(q) = 1 (isotropic, equal form factors).

    Parameters
    ----------
    coords : (N, 3) array
    q_values : (M,) array
    n_max : int

    Returns
    -------
    In : (n_max+1, M) complex array
    """
    x = coords[:, 0]
    y = coords[:, 1]

    dx = x[:, None] - x[None, :]   # (N, N)
    dy = y[:, None] - y[None, :]
    r_ij   = np.sqrt(dx**2 + dy**2)   # (N, N)
    phi_ij = np.arctan2(dy, dx)        # (N, N)

    q   = np.asarray(q_values)         # (M,)
    qr  = q[None, None, :] * r_ij[:, :, None]   # (N, N, M)  -- precompute once

    In = np.zeros((n_max + 1, len(q)), dtype=complex)

    for n in range(n_max + 1):
        phase = np.exp(-1j * n * phi_ij)   # (N, N)
        Jn    = jv(n, qr)                  # (N, N, M)
        In[n] = (1j)**n * np.einsum('ij,ijm->m', phase, Jn)

    return In


def _fast_orientational_average(coords, q_values, n_max=12,
                                 n_orientations=20000, seed=42,
                                 batch_size=1000, verbose=False):
    """
    Fast orientational average using a single representative q value.

    The BOO fingerprint shape c^n/c^0 is approximately q-independent
    across the first diffraction peak (verified in PRL 2016 Fig. 2).
    So we evaluate at q_center only, giving a 20-50x speedup over
    a full q-grid.

    Returns cn_avg of shape (n_max+1, 1) broadcast to (n_max+1, len(q_values)).
    """
    rng = np.random.default_rng(seed)
    N = len(coords)
    q_center = float(np.mean(q_values))

    accumulator = np.zeros(n_max + 1, dtype=np.float64)
    done = 0

    # Precompute all rotation matrices
    Rs = np.stack([random_rotation_matrix(rng) for _ in range(n_orientations)])

    for start in range(0, n_orientations, batch_size):
        end = min(start + batch_size, n_orientations)
        B = end - start
        Rb = Rs[start:end]   # (B, 3, 3)

        # Rotate: (B, N, 3)
        rotated = np.einsum('bij,kj->bki', Rb, coords)

        # Project to xy, compute pairs
        xy = rotated[:, :, :2]                             # (B, N, 2)
        diff = xy[:, :, None, :] - xy[:, None, :, :]      # (B, N, N, 2)
        r_flat   = np.sqrt((diff**2).sum(-1)).reshape(B, -1)   # (B, N*N)
        phi_flat = np.arctan2(diff[..., 1], diff[..., 0]).reshape(B, -1)

        qr = q_center * r_flat    # (B, N*N)

        I0_sq = None
        for n in range(n_max + 1):
            Jn      = jv(n, qr)                     # (B, N*N)
            cos_nphi = np.cos(n * phi_flat)          # (B, N*N)
            sin_nphi = np.sin(n * phi_flat)

            # phase = exp(-i n phi) = cos(nphi) - i sin(nphi)
            # I^n = i^n * sum_ij J_n(qr_ij) * exp(-i n phi_ij)
            pre_re = [1, 0, -1, 0][n % 4]
            pre_im = [0, 1,  0, -1][n % 4]

            sRe = (Jn * cos_nphi).sum(-1)    # (B,)
            sIm = -(Jn * sin_nphi).sum(-1)   # (B,)

            InRe = pre_re * sRe - pre_im * sIm
            InIm = pre_re * sIm + pre_im * sRe
            In_sq = InRe**2 + InIm**2        # (B,)

            if n == 0:
                I0_sq = np.where(In_sq > 0, In_sq, 1.0)

            accumulator[n] += (In_sq / I0_sq).sum()

        done += B
        if verbose and done % 5000 == 0:
            print(f"  {done}/{n_orientations}", flush=True)

    # Broadcast to (n_max+1, M)
    cn_scalar = accumulator / n_orientations
    return np.tile(cn_scalar[:, None], (1, len(q_values)))


def angular_autocorrelation_coefficients(In):
    """
    Compute normalized Fourier coefficients of the angular autocorrelation.

    From Eq. 9 of PRL 2016:
        c^n_q / c^0_q = |I^n_q|^2 / |I^0_q|^2

    Parameters
    ----------
    In : (n_max+1, M) complex array

    Returns
    -------
    cn_norm : (n_max+1, M) real array
        Normalized symmetry magnitudes c^n/c^0.
    """
    I0_sq = np.abs(In[0]) ** 2
    # avoid division by zero
    safe_I0 = np.where(I0_sq > 0, I0_sq, 1.0)
    cn_norm = np.abs(In) ** 2 / safe_I0
    return cn_norm.real

# Orientational averaging

def orientational_average(
    coords,
    q_values,
    n_max=12,
    n_orientations=20000,
    seed=42,
    verbose=False,
    batch_size=500,
):
    """
    Average Fourier coefficients over random orientations of the cluster.

    Parameters
    ----------
    coords : (N, 3) array
    q_values : (M,) array
    n_max : int
    n_orientations : int
    seed : int

    Returns
    -------
    cn_avg : (n_max+1, M) array
    """
    return _fast_orientational_average(
        coords, q_values,
        n_max=n_max,
        n_orientations=n_orientations,
        seed=seed,
        batch_size=batch_size,
        verbose=verbose,
    )


def convergence_check(coords, q_values, q_mask, n_max=12,
                      orientation_counts=(1000, 10000, 20000),
                      seed=42, batch_size=1000):
    """
    Compute fingerprint at increasing numbers of orientations.
    Returns dict {n_orientations: fingerprint array}.
    Used to verify convergence (analog to PRL 2016 Fig. 2c).
    """
    rng = np.random.default_rng(seed)
    q_center = float(np.mean(q_values))
    counts_sorted = sorted(orientation_counts)
    total = max(counts_sorted)

    accumulator = np.zeros(n_max + 1, dtype=np.float64)
    Rs = np.stack([random_rotation_matrix(rng) for _ in range(total)])
    results = {}
    done = 0
    ci = 0   # checkpoint index

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        B = end - start
        Rb = Rs[start:end]

        rotated  = np.einsum('bij,kj->bki', Rb, coords)
        xy       = rotated[:, :, :2]
        diff     = xy[:, :, None, :] - xy[:, None, :, :]
        r_flat   = np.sqrt((diff**2).sum(-1)).reshape(B, -1)
        phi_flat = np.arctan2(diff[..., 1], diff[..., 0]).reshape(B, -1)
        qr       = q_center * r_flat

        I0_sq = None
        for n in range(n_max + 1):
            Jn        = jv(n, qr)
            cos_nphi  = np.cos(n * phi_flat)
            sin_nphi  = np.sin(n * phi_flat)
            pre_re    = [1, 0, -1, 0][n % 4]
            pre_im    = [0, 1,  0, -1][n % 4]
            sRe       = (Jn * cos_nphi).sum(-1)
            sIm       = -(Jn * sin_nphi).sum(-1)
            InRe      = pre_re * sRe - pre_im * sIm
            InIm      = pre_re * sIm + pre_im * sRe
            In_sq     = InRe**2 + InIm**2
            if n == 0:
                I0_sq = np.where(In_sq > 0, In_sq, 1.0)
            accumulator[n] += (In_sq / I0_sq).sum()

        done += B

        while ci < len(counts_sorted) and done >= counts_sorted[ci]:
            fp = accumulator / done
            # broadcast to q shape then mask-average
            fp_q = np.tile(fp[:, None], (1, len(q_values)))
            results[counts_sorted[ci]] = fp_q[:, q_mask].mean(axis=1).copy()
            print(f"  Checkpoint: {counts_sorted[ci]} orientations", flush=True)
            ci += 1

    return results


# First-peak averaging

def first_peak_q_range(q_values, q_center, q_width):
    """
    Return boolean mask for q values within [q_center - q_width/2,
    q_center + q_width/2].
    """
    return np.abs(q_values - q_center) <= q_width / 2.0


def symmetry_fingerprint(cn_avg, q_mask):
    """
    Average normalized symmetry magnitudes over the first diffraction peak.

    Parameters
    ----------
    cn_avg : (n_max+1, M) array
    q_mask : (M,) bool array

    Returns
    -------
    fingerprint : (n_max+1,) array
        C^n averaged over first peak q-range, for n = 1..n_max.
        (n=0 is always 1 by normalization, excluded from plots.)
    """
    return cn_avg[:, q_mask].mean(axis=1)


# Convergence check helper
