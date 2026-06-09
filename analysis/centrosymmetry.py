"""
Local centrosymmetry and structural anisotropy from scanning SAXS patterns.

Implements two parameters from Liu et al., Science Advances 8, eabn0681 (2022):

1. CENTROSYMMETRY RATIO: Sigma c_{2n+2} / Sigma c_{2n+1}  (Eq. after Fig. 1C)
   Probes the degree of local inversion symmetry in particle arrangements.
   In centrosymmetric configurations, dynamical diffraction produces no odd-order
   Fourier coefficients, so c_{2n+1} -> 0 and the ratio -> large.
   Non-centrosymmetric arrangements produce nonzero odd terms, reducing the ratio.

2. ANISOTROPY TENSOR: epsilon(phi) = epsilon_xx cos²phi + epsilon_xy cos(phi)sin(phi)
                                    + epsilon_yy sin²phi   (Eq. 5)
   Fit the azimuthal variation of q_max (peak position in each angular arc)
   to extract the in-plane distortion tensor components.
   Normal anisotropy: epsilon_n = (epsilon_xx + epsilon_yy) / 2  (dilatation)
   Shear anisotropy:  epsilon_s = epsilon_xy                     (volume-conserving)

Reference: Liu et al., Sci. Adv. 8, eabn0681 (2022)
"""

import numpy as np
from scipy.optimize import curve_fit


# Centrosymmetry ratio


def centrosymmetry_ratio(cn_coeffs, n_max=11):
    """
    Compute the centrosymmetry ratio Sigma c_{2n+2} / Sigma c_{2n+1}.

    Parameters
    ----------
    cn_coeffs : (N_max+1,) array
        Fourier coefficients c_n/c_0 for n = 0, 1, 2, ..., N_max.
        As returned by symmetry_fingerprint() in diffraction.py.
    n_max : int
        Maximum n to include. Default 11 gives pairs up to (c_12, c_11).

    Returns
    -------
    ratio : float
        Sigma c_{2n+2} / Sigma c_{2n+1} for n = 0..4 (even: 2,4,6,8,10,12;
        odd: 1,3,5,7,9,11).
        Large ratio = more centrosymmetric.
        Small ratio = more asymmetric / non-centrosymmetric.
    sum_even : float
    sum_odd  : float
    """
    cn = np.asarray(cn_coeffs)

    # Even terms: c_2, c_4, c_6, c_8, c_10, c_12
    even_indices = np.arange(2, min(n_max + 1, len(cn)), 2)
    # Odd terms:  c_1, c_3, c_5, c_7, c_9, c_11
    odd_indices  = np.arange(1, min(n_max + 1, len(cn)), 2)

    sum_even = np.sum(np.abs(cn[even_indices]))
    sum_odd  = np.sum(np.abs(cn[odd_indices]))

    ratio = sum_even / max(sum_odd, 1e-12)
    return ratio, sum_even, sum_odd


def centrosymmetry_ratio_from_pattern(pattern_2d, q_values, phi_values,
                                       q_min, q_max_range, n_max=12):
    """
    Compute centrosymmetry ratio directly from a 2D diffraction pattern I(q, phi).

    Parameters
    ----------
    pattern_2d : (N_q, N_phi) array
        Diffraction intensity on a polar grid.
    q_values   : (N_q,) array
    phi_values : (N_phi,) array
        Azimuthal angles in radians.
    q_min, q_max_range : float
        q range for the first diffraction peak averaging.
    n_max : int

    Returns
    -------
    ratio : float
    cn    : (n_max+1,) array of Fourier coefficients
    """
    # Select q range
    mask = (q_values >= q_min) & (q_values <= q_max_range)
    I_peak = pattern_2d[mask, :]   # (N_q_sel, N_phi)

    # Average over q range
    I_avg = I_peak.mean(axis=0)    # (N_phi,)

    # Fourier coefficients via DFT
    dphi = phi_values[1] - phi_values[0]
    cn = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        cn[n] = np.abs(
            np.sum(I_avg * np.exp(-1j * n * phi_values)) * dphi / (2 * np.pi)
        )

    # Normalize by c_0
    cn = cn / max(cn[0], 1e-12)

    ratio, _, _ = centrosymmetry_ratio(cn, n_max=n_max)
    return ratio, cn


# Anisotropy tensor fitting


def anisotropy_model(phi, eps_xx, eps_xy, eps_yy):
    """
    Anisotropy function epsilon(phi) = eps_xx cos²phi + eps_xy cos(phi)sin(phi)
                                     + eps_yy sin²phi

    This is Eq. 5 of Sci Adv 2022. It measures how much q_max deviates from
    the isotropic average as a function of azimuthal angle phi.

    Parameters
    ----------
    phi : array of azimuthal angles (radians)

    Returns
    -------
    epsilon(phi) : array
    """
    return (eps_xx * np.cos(phi)**2
            + eps_xy * np.cos(phi) * np.sin(phi)
            + eps_yy * np.sin(phi)**2)


def fit_anisotropy(phi_values, q_max_vs_phi, q0=None):
    """
    Fit the distortion tensor from q_max(phi) measurements.

    The anisotropy is defined as:
        epsilon(phi) = (q0 - q_max(phi)) / q_max(phi)

    which is then fitted to eps_xx cos²phi + eps_xy cosphi sinphi + eps_yy sin²phi.

    Parameters
    ----------
    phi_values    : (N_phi,) array of azimuthal angles (radians)
    q_max_vs_phi  : (N_phi,) array of peak q position in each angular arc
    q0            : float or None
        Reference isotropic q_max. If None, uses mean of q_max_vs_phi.

    Returns
    -------
    eps_n : float  — normal anisotropy = (eps_xx + eps_yy) / 2
    eps_s : float  — shear anisotropy  = eps_xy
    tensor : dict  {'xx': eps_xx, 'xy': eps_xy, 'yy': eps_yy}
    fit_curve : (N_phi,) array — fitted epsilon(phi)
    """
    if q0 is None:
        q0 = np.mean(q_max_vs_phi)

    epsilon = (q0 - q_max_vs_phi) / np.where(q_max_vs_phi > 0, q_max_vs_phi, 1.0)

    try:
        popt, pcov = curve_fit(
            anisotropy_model, phi_values, epsilon,
            p0=[0.0, 0.0, 0.0],
            maxfev=2000,
        )
        eps_xx, eps_xy, eps_yy = popt
    except RuntimeError:
        eps_xx = eps_xy = eps_yy = 0.0

    eps_n = (eps_xx + eps_yy) / 2.0   # normal (dilatation)
    eps_s = eps_xy                      # shear

    fit_curve = anisotropy_model(phi_values, eps_xx, eps_xy, eps_yy)

    tensor = {'xx': float(eps_xx), 'xy': float(eps_xy), 'yy': float(eps_yy)}
    return float(eps_n), float(eps_s), tensor, fit_curve


def extract_qmax_arcs(pattern_2d, q_values, phi_values,
                       q_min, q_max_range, n_arcs=24):
    """
    Divide the first diffraction ring into n_arcs angular sectors and find
    q_max (center of mass of intensity) in each arc.

    This is the experimental procedure of Sci Adv 2022 Materials and Methods.

    Parameters
    ----------
    pattern_2d : (N_q, N_phi) array
    q_values   : (N_q,) array
    phi_values : (N_phi,) array
    q_min, q_max_range : float  — q range of first peak
    n_arcs     : int  — number of angular sectors (paper uses 24)

    Returns
    -------
    phi_centers : (n_arcs,) array — arc center angles
    qmax_arcs   : (n_arcs,) array — q_max per arc
    """
    q_mask = (q_values >= q_min) & (q_values <= q_max_range)
    q_sel  = q_values[q_mask]

    arc_width = 2 * np.pi / n_arcs
    phi_centers = np.linspace(0, 2 * np.pi - arc_width, n_arcs) + arc_width / 2
    qmax_arcs   = np.zeros(n_arcs)

    phi_wrapped = phi_values % (2 * np.pi)

    for i, phi_c in enumerate(phi_centers):
        phi_lo = (phi_c - arc_width / 2) % (2 * np.pi)
        phi_hi = (phi_c + arc_width / 2) % (2 * np.pi)

        if phi_lo < phi_hi:
            arc_mask = (phi_wrapped >= phi_lo) & (phi_wrapped < phi_hi)
        else:
            arc_mask = (phi_wrapped >= phi_lo) | (phi_wrapped < phi_hi)

        if arc_mask.sum() == 0:
            qmax_arcs[i] = q_sel.mean()
            continue

        # Average intensity over arc
        I_arc = pattern_2d[q_mask, :][:, arc_mask].mean(axis=1)  # (N_q_sel,)

        # Center of mass
        total = I_arc.sum()
        if total > 0:
            qmax_arcs[i] = np.sum(q_sel * I_arc) / total
        else:
            qmax_arcs[i] = q_sel.mean()

    return phi_centers, qmax_arcs


# Synthetic diffraction pattern generator


def make_synthetic_pattern(cluster_coords, q_values, phi_values,
                            n_arcs=360, distortion=None, noise=0.02,
                            seed=42):
    """
    Generate a synthetic 2D diffraction pattern I(q, phi) from cluster coords.

    Uses the kinematical approximation: I(q,phi) = |F(q,phi)|^2 where
    F is the structure factor. Optionally applies an anisotropic distortion
    (simulates a strained glass region).

    Parameters
    ----------
    cluster_coords : (N, 3) array
    q_values       : (N_q,) array
    phi_values     : (N_phi,) array  — azimuthal angles
    distortion     : dict or None
        If given, {'eps_xx': ..., 'eps_xy': ..., 'eps_yy': ...}
        applies a structural distortion to the cluster before computing.
    noise          : float  — relative Gaussian noise level
    seed           : int

    Returns
    -------
    pattern : (N_q, N_phi) array
    """
    rng = np.random.default_rng(seed)
    coords = cluster_coords.copy()

    # Apply distortion if given
    if distortion is not None:
        e = distortion
        strain_matrix = np.array([
            [1 + e.get('eps_xx', 0), e.get('eps_xy', 0) / 2, 0],
            [e.get('eps_xy', 0) / 2, 1 + e.get('eps_yy', 0), 0],
            [0, 0, 1],
        ])
        coords = coords @ strain_matrix.T

    x = coords[:, 0]
    y = coords[:, 1]
    N = len(coords)

    pattern = np.zeros((len(q_values), len(phi_values)))

    for pi, phi in enumerate(phi_values):
        qx = q_values * np.cos(phi)   # (N_q,)
        qy = q_values * np.sin(phi)

        # Structure factor F(q, phi) = sum_i exp(i q . r_i)
        # phase: (N_q, N_atoms)
        phase = np.exp(1j * (qx[:, None] * x[None, :] + qy[:, None] * y[None, :]))
        F = phase.sum(axis=1)          # (N_q,)
        pattern[:, pi] = np.abs(F)**2

    # Add noise
    if noise > 0:
        pattern += rng.normal(0, noise * pattern.mean(), pattern.shape)
        pattern = np.clip(pattern, 0, None)

    return pattern


# Spatial map simulation (aging/deformation analog)


def simulate_spatial_maps(fingerprints_dict, grid_size=20,
                           centrosym_field=None, aniso_field=None,
                           seed=42):
    """
    Simulate a spatial map of centrosymmetry and anisotropy parameters
    across a glass specimen grid.

    This produces the kind of 2D maps shown in Sci Adv 2022 Figs. 2 and 4.

    Parameters
    ----------
    fingerprints_dict : dict {name: fp_array}
    grid_size  : int — NxN spatial grid
    centrosym_field : (N, N) array or None
        Pre-defined centrosymmetry variation. If None, generates random.
    aniso_field : (N, N, 2) array or None
        Pre-defined (eps_n, eps_s) fields. If None, generates random.
    seed : int

    Returns
    -------
    maps : dict with keys 'centrosymmetry', 'eps_n', 'eps_s', 'stability'
        Each is a (grid_size, grid_size) array.
    """
    rng = np.random.default_rng(seed)
    N = grid_size

    if centrosym_field is None:
        # Smooth random field via Gaussian blur analog
        raw = rng.standard_normal((N, N))
        centrosym_field = _smooth_field(raw, sigma=2.0)

    if aniso_field is None:
        raw_n = rng.standard_normal((N, N))
        raw_s = rng.standard_normal((N, N))
        eps_n = _smooth_field(raw_n, sigma=2.0) * 0.015
        eps_s = _smooth_field(raw_s, sigma=2.0) * 0.010
        aniso_field = np.stack([eps_n, eps_s], axis=-1)

    # Centrosymmetry: normalize to realistic range
    cs = centrosym_field
    cs = (cs - cs.mean()) / (cs.std() + 1e-9)
    cs = cs * 0.15 + 1.0    # mean ~ 1.0, std ~ 0.15 (paper values)

    eps_n = aniso_field[:, :, 0]
    eps_s = aniso_field[:, :, 1]

    # Stability: positively correlated with centrosymmetry + noise
    stability = cs * 0.4 + rng.standard_normal((N, N)) * 0.05
    stability = (stability - stability.min()) / (stability.max() - stability.min())
    stability = stability * 4 + 18.5   # scale to realistic C(t) range ~19-23

    return {
        'centrosymmetry': cs,
        'eps_n': eps_n,
        'eps_s': eps_s,
        'stability': stability,
    }


def simulate_aging(maps_young, grid_size=20, seed=43):
    """
    Simulate aging effect on structural maps.

    Aging (Sci Adv 2022 Fig. 2):
    - Stability increases
    - Centrosymmetry decreases (polyhedra relax to lower symmetry)
    - Anisotropy (eps_n) broadens and develops high-side shoulder
    - Spatial correlation lengths increase
    """
    rng = np.random.default_rng(seed)
    N = grid_size

    cs_young = maps_young['centrosymmetry']
    en_young = maps_young['eps_n']
    es_young = maps_young['eps_s']

    # Aging: shift centrosymmetry down, increase correlation length
    raw = rng.standard_normal((N, N))
    cs_aged = _smooth_field(raw, sigma=3.5) * 0.12 + 0.88   # lower mean, longer corr

    # Normal anisotropy increases and develops high-side tail
    raw_n = rng.standard_normal((N, N))
    eps_n_aged = _smooth_field(raw_n, sigma=3.5) * 0.020 + 0.005

    raw_s = rng.standard_normal((N, N))
    eps_s_aged = _smooth_field(raw_s, sigma=2.5) * 0.010

    stability_aged = cs_aged * 0.5 + rng.standard_normal((N, N)) * 0.04
    stability_aged = (stability_aged - stability_aged.min()) / \
                     (stability_aged.max() - stability_aged.min())
    stability_aged = stability_aged * 3 + 19.5   # higher mean than young

    return {
        'centrosymmetry': cs_aged,
        'eps_n': eps_n_aged,
        'eps_s': eps_s_aged,
        'stability': stability_aged,
    }


def simulate_deformation(maps_fresh, grid_size=20, seed=44):
    """
    Simulate deformation effect on structural maps.

    Deformation (Sci Adv 2022 Fig. 4):
    - Stability distribution broadens, develops low-stability tail
    - Shear bands appear: elongated low-centrosymmetry, high-eps_n bands
    - Spatial correlation lengths become very long
    """
    rng = np.random.default_rng(seed)
    N = grid_size

    # Base field
    raw = rng.standard_normal((N, N))
    cs_base = _smooth_field(raw, sigma=2.5) * 0.14 + 0.95

    # Add shear band: diagonal stripe of low centrosymmetry
    band = _make_shear_band(N, angle=45, width=3, seed=seed)
    cs_deformed = cs_base - band * 0.25

    # eps_n large in the band
    eps_n_deformed = _smooth_field(rng.standard_normal((N, N)), sigma=3.0) * 0.015
    eps_n_deformed += band * 0.03

    eps_s_deformed = _smooth_field(rng.standard_normal((N, N)), sigma=2.0) * 0.008

    stability_def = cs_deformed * 0.35 + rng.standard_normal((N, N)) * 0.06
    stability_def = (stability_def - stability_def.min()) / \
                    (stability_def.max() - stability_def.min())
    stability_def = stability_def * 5 + 17.0   # broader range than aging

    return {
        'centrosymmetry': cs_deformed,
        'eps_n': eps_n_deformed,
        'eps_s': eps_s_deformed,
        'stability': stability_def,
    }


# Internal helpers

def _smooth_field(field, sigma=2.0):
    """Smooth a 2D field with a Gaussian kernel (manual convolution)."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(field, sigma=sigma)


def _make_shear_band(N, angle=45, width=3, seed=42):
    """Generate a diagonal shear band mask in an NxN grid."""
    rng = np.random.default_rng(seed)
    band = np.zeros((N, N))
    angle_rad = np.radians(angle)
    for i in range(N):
        for j in range(N):
            # Distance from diagonal line
            dist = abs((j - i) * np.cos(angle_rad))
            if dist < width:
                band[i, j] = np.exp(-0.5 * (dist / (width / 2))**2)
    return band
