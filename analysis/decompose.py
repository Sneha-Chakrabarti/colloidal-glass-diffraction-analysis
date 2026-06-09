"""
NNLS decomposition of average symmetry spectra into polyhedral contributions.

Implements Eq. 2 of Liu et al., PNAS 114, 10344 (2017):

    C^n = [C^n_FCC, C^n_BCC, C^n_ICO, C^n_RAN]^T @ [f_FCC, f_BCC, f_ICO, f_RAN]

where C^n is the measured average symmetry spectrum (n=2..12),
C^n_cluster are the precomputed projected BOO fingerprints,
C^n_RAN is a constant (flat) spectrum representing structural noise,
and f_cluster are the nonnegative weights found by NNLS.

Reference: Liu et al., PNAS 114, 10344 (2017) and
           Liu et al., PRL 116, 205501 (2016)
"""

import numpy as np
from scipy.optimize import nnls


# Basis construction


def build_basis(fingerprints, include_ran=True, n_min=2, n_max=12):
    """
    Construct the basis matrix for NNLS fitting.

    Parameters
    ----------
    fingerprints : dict
        {cluster_name: fp_array (n_max+1,)} from orientational averaging.
        Keys should include at least 'FCC', 'BCC', 'ICO'.
    include_ran : bool
        If True, add a constant (flat) RAN component.
    n_min, n_max : int
        Symmetry range to use in the fit.

    Returns
    -------
    basis : (n_syms, n_components) array
        Each column is one basis vector (fingerprint for n=n_min..n_max).
    labels : list of str
        Component labels corresponding to columns.
    """
    n_syms = n_max - n_min + 1
    labels = []
    columns = []

    for name, fp in fingerprints.items():
        col = fp[n_min:n_max + 1]
        columns.append(col)
        labels.append(name)

    if include_ran:
        # RAN = constant offset = mean of all fingerprints, or just 1.0
        ran = np.ones(n_syms)
        ran = ran / ran.mean()   # normalize
        columns.append(ran)
        labels.append('RAN')

    basis = np.column_stack(columns)   # (n_syms, n_components)
    return basis, labels


# NNLS fit

def nnls_decompose(spectrum, basis, labels):
    """
    Fit a measured symmetry spectrum as a nonnegative linear combination
    of basis vectors.

    Parameters
    ----------
    spectrum : (n_syms,) array
        Measured average symmetry magnitudes C^n for n=n_min..n_max.
    basis : (n_syms, n_components) array
    labels : list of str

    Returns
    -------
    weights : dict {label: weight}
    fit : (n_syms,) array  — reconstructed spectrum
    residual : float — ||spectrum - fit||^2
    """
    w, res = nnls(basis, spectrum)
    fit = basis @ w
    residual = float(np.sum((spectrum - fit) ** 2))
    weights = {label: float(wt) for label, wt in zip(labels, w)}
    return weights, fit, residual


def nnls_decompose_with_errors(spectrum, basis, labels,
                                sigma=None, delta_chi2=1.0):
    """
    NNLS fit with confidence ranges estimated by varying each weight
    until chi^2 increases by delta_chi2 (method of PNAS 2017).

    Parameters
    ----------
    spectrum : (n_syms,) array
    basis : (n_syms, n_components) array
    labels : list of str
    sigma : (n_syms,) array or None
        Per-point uncertainties. If None, assumes uniform sigma=1.
    delta_chi2 : float
        chi^2 increase used to define confidence range.

    Returns
    -------
    weights : dict {label: (best, low, high)}
    fit : (n_syms,) array
    chi2_reduced : float
    """
    if sigma is None:
        sigma = np.ones_like(spectrum)

    # Weighted NNLS: divide both sides by sigma
    spectrum_w = spectrum / sigma
    basis_w    = basis / sigma[:, None]

    w_best, _ = nnls(basis_w, spectrum_w)
    fit = basis @ w_best
    n_syms = len(spectrum)
    n_free = max(1, n_syms - np.sum(w_best > 1e-10))
    chi2_best = float(np.sum(((spectrum - fit) / sigma) ** 2))
    chi2_reduced = chi2_best / max(1, n_syms - 1)

    # Confidence ranges: scan each weight individually
    errors = {}
    for i, label in enumerate(labels):
        w_lo = _scan_weight(w_best, i, basis_w, spectrum_w,
                            chi2_best, delta_chi2, direction='down')
        w_hi = _scan_weight(w_best, i, basis_w, spectrum_w,
                            chi2_best, delta_chi2, direction='up')
        errors[label] = (float(w_best[i]), float(w_lo), float(w_hi))

    return errors, fit, chi2_reduced


def _scan_weight(w_best, idx, basis_w, spectrum_w,
                 chi2_best, delta_chi2, direction='up',
                 n_steps=200, factor=2.0):
    """Scan weight[idx] to find where chi^2 increases by delta_chi2."""
    w = w_best.copy()
    w_range = max(w_best[idx] * factor, 0.05)
    if direction == 'up':
        candidates = np.linspace(w_best[idx], w_best[idx] + w_range, n_steps)
    else:
        candidates = np.linspace(max(0, w_best[idx] - w_range), w_best[idx], n_steps)
        candidates = candidates[::-1]

    for wc in candidates:
        w[idx] = wc
        residual = basis_w @ w - spectrum_w
        chi2 = float(np.sum(residual**2))
        if chi2 - chi2_best >= delta_chi2:
            return wc

    return candidates[-1]


# Synthetic ensemble generator (for testing / PNAS Fig 4 reproduction)


def simulate_mixed_ensemble(fingerprints_dict, mixture, n_orientations=5000,
                             noise_level=0.01, seed=42):
    """
    Simulate the average symmetry spectrum for a known mixture of clusters.

    Parameters
    ----------
    fingerprints_dict : dict {name: fp_array}
    mixture : dict {name: fraction}
        Cluster fractions (should sum to ~1, or will be normalized).
        Use 'RAN' for random noise component.
    n_orientations : int
        Simulated measurement statistics (controls noise floor).
    noise_level : float
        Additional Gaussian noise std relative to signal.
    seed : int

    Returns
    -------
    spectrum : (n_syms,) array  n=2..12
    true_weights : dict
    """
    rng = np.random.default_rng(seed)
    total = sum(mixture.values())
    norm_mix = {k: v / total for k, v in mixture.items()}

    spectrum = np.zeros(11)  # n=2..12
    for name, frac in norm_mix.items():
        if name == 'RAN':
            spectrum += frac * np.ones(11)
        else:
            spectrum += frac * fingerprints_dict[name][2:13]

    # Add realistic noise (Poisson-like, scales as 1/sqrt(N))
    noise_sigma = noise_level * spectrum.mean() + 1.0 / np.sqrt(n_orientations)
    spectrum += rng.normal(0, noise_sigma, size=spectrum.shape)
    spectrum = np.clip(spectrum, 0, None)

    return spectrum, norm_mix


# Normalisation helper

def normalize_weights(weights_dict):
    """
    Normalize weights so they sum to 1.
    Returns new dict and total.
    """
    total = sum(weights_dict.values())
    if total == 0:
        return weights_dict, 0.0
    return {k: v / total for k, v in weights_dict.items()}, total
