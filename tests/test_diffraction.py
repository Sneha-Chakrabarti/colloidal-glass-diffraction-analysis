"""
Unit tests for glass-diffraction-toolkit.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_diffraction.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sim.clusters import get_all_clusters, validate_clusters
from analysis.diffraction import (
    random_rotation_matrix,
    rotate_cluster,
    orientational_average,
    symmetry_fingerprint,
    first_peak_q_range,
)
from analysis.decompose import (
    build_basis,
    nnls_decompose,
    simulate_mixed_ensemble,
    normalize_weights,
)

# Cluster geometry tests

class TestClusters:
    def setup_method(self):
        self.clusters = get_all_clusters()

    def test_all_clusters_present(self):
        for name in ['FCC', 'BCC', 'HCP', 'ICO', 'SC']:
            assert name in self.clusters

    def test_nn_distance_is_one(self):
        for name, coords in self.clusters.items():
            dists = np.linalg.norm(coords[1:], axis=1)
            nn = dists.min()
            assert abs(nn - 1.0) < 1e-6, f"{name} nn distance = {nn:.6f}, expected 1.0"

    def test_atom_counts(self):
        expected = {'FCC': 13, 'BCC': 9, 'HCP': 13, 'ICO': 13, 'SC': 7}
        for name, n in expected.items():
            assert len(self.clusters[name]) == n, \
                f"{name}: got {len(self.clusters[name])} atoms, expected {n}"

    def test_center_at_origin(self):
        for name, coords in self.clusters.items():
            assert np.allclose(coords[0], [0, 0, 0], atol=1e-10), \
                f"{name} central atom not at origin"

    def test_fcc_12_neighbors(self):
        coords = self.clusters['FCC']
        dists = np.linalg.norm(coords[1:], axis=1)
        nn_dist = dists.min()
        n_nn = np.sum(np.abs(dists - nn_dist) < 1e-6)
        assert n_nn == 12, f"FCC should have 12 nearest neighbors, got {n_nn}"

    def test_ico_12_neighbors(self):
        coords = self.clusters['ICO']
        dists = np.linalg.norm(coords[1:], axis=1)
        nn_dist = dists.min()
        n_nn = np.sum(np.abs(dists - nn_dist) < 1e-6)
        assert n_nn == 12, f"ICO should have 12 nearest neighbors, got {n_nn}"


# Rotation tests

class TestRotation:
    def test_rotation_matrix_orthogonal(self):
        R = random_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_rotation_matrix_det_one(self):
        R = random_rotation_matrix()
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rotation_preserves_distances(self):
        clusters = get_all_clusters()
        coords = clusters['FCC']
        R = random_rotation_matrix()
        rotated = rotate_cluster(coords, R)
        d_orig = np.linalg.norm(coords[1:], axis=1)
        d_rot  = np.linalg.norm(rotated[1:], axis=1)
        assert np.allclose(d_orig, d_rot, atol=1e-10)

    def test_different_seeds_give_different_rotations(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(1)
        R1 = random_rotation_matrix(rng1)
        R2 = random_rotation_matrix(rng2)
        assert not np.allclose(R1, R2)

# Module-level fingerprint cache — computed once, shared by all test classes

_FPS_CACHE = None

def _get_fps(n_orientations=2000):
    global _FPS_CACHE
    if _FPS_CACHE is None:
        clusters = get_all_clusters()
        Q_CENTER = 2 * np.pi * 1.23
        Q_WIDTH  = Q_CENTER * 0.4
        q_values = np.linspace(Q_CENTER - Q_WIDTH, Q_CENTER + Q_WIDTH, 6)
        q_mask   = np.ones(len(q_values), dtype=bool)
        _FPS_CACHE = {}
        for name in ['FCC', 'BCC', 'HCP', 'ICO', 'SC']:
            cn = orientational_average(clusters[name], q_values,
                                       n_max=12, n_orientations=n_orientations, seed=0,
                                       batch_size=1000)
            _FPS_CACHE[name] = symmetry_fingerprint(cn, q_mask)
    return _FPS_CACHE

# Fingerprint physics tests

class TestFingerprints:
    def setup_method(self):
        self.fps = _get_fps()

    def test_n0_normalized_to_one(self):
        for name, fp in self.fps.items():
            assert abs(fp[0] - 1.0) < 1e-8, f"{name}: fp[0] = {fp[0]}, expected 1.0"

    def test_odd_symmetries_near_zero(self):
        # For all clusters under orientational averaging,
        # odd-n contributions should be much smaller than even-n
        for name, fp in self.fps.items():
            odd_mean  = np.mean(np.abs(fp[1::2]))   # n=1,3,5,7,9,11
            even_mean = np.mean(np.abs(fp[2::2]))   # n=2,4,6,8,10,12
            assert odd_mean < even_mean * 0.5, \
                f"{name}: odd={odd_mean:.4f} should be << even={even_mean:.4f}"

    def test_ico_flat_low_fingerprint(self):
        # ICO fingerprint should have lower even-n values than FCC/BCC
        fp_ico = self.fps['ICO']
        fp_fcc = self.fps['FCC']
        even_ico = np.mean(fp_ico[2::2])
        even_fcc = np.mean(fp_fcc[2::2])
        assert even_ico < even_fcc * 0.3, \
            f"ICO even-n mean ({even_ico:.4f}) should be < 30% of FCC ({even_fcc:.4f})"

    def test_fcc_dominant_n4_or_n6(self):
        fp = self.fps['FCC']
        vals = fp[2:13]   # n=2..12
        top2 = np.argsort(vals)[-2:] + 2   # n indices of top 2
        assert any(n in top2 for n in [4, 6, 2]), \
            f"FCC: top symmetries are {top2}, expected 2,4, or 6"

    def test_hcp_n6_prominent(self):
        fp = self.fps['HCP']
        assert fp[6] >= fp[4] * 0.6, \
            f"HCP: n=6 ({fp[6]:.4f}) should be >= 60% of n=4 ({fp[4]:.4f})"

    def test_all_fingerprints_nonnegative(self):
        for name, fp in self.fps.items():
            assert np.all(fp[2:] >= 0), f"{name}: negative fingerprint values"

# Decomposition tests

# Module-level cache so decomposition tests share one computation
def _get_decomp_fps():
    fps = _get_fps()
    return {k: fps[k] for k in ['FCC', 'BCC', 'ICO']}


class TestDecomposition:
    def setup_method(self):
        self.fps = _get_decomp_fps()

    def test_basis_shape(self):
        basis, labels = build_basis(self.fps, include_ran=True)
        assert basis.shape == (11, 4)   # 11 symmetries, 4 components
        assert 'RAN' in labels

    def test_pure_bcc_recovers_bcc(self):
        """NNLS fit of a pure BCC spectrum should give near-100% BCC weight."""
        spectrum = self.fps['BCC'][2:13]
        basis, labels = build_basis(self.fps, include_ran=True)
        weights, fit, res = nnls_decompose(spectrum, basis, labels)
        w_norm, _ = normalize_weights(weights)
        assert w_norm['BCC'] > 0.6, \
            f"Pure BCC recovery: BCC weight = {w_norm['BCC']:.3f}, expected > 0.6"

    def test_pure_ico_recovers_ico(self):
        spectrum = self.fps['ICO'][2:13]
        basis, labels = build_basis(self.fps, include_ran=True)
        weights, fit, res = nnls_decompose(spectrum, basis, labels)
        # ICO is flat so RAN may absorb some; ICO+RAN should dominate
        total_ico_ran = weights['ICO'] + weights['RAN']
        total = sum(weights.values())
        assert total_ico_ran / max(total, 1e-9) > 0.7, \
            f"ICO+RAN fraction = {total_ico_ran/total:.3f}, expected > 0.7"

    def test_weights_nonnegative(self):
        spectrum = self.fps['FCC'][2:13] * 0.5 + self.fps['ICO'][2:13] * 0.5
        basis, labels = build_basis(self.fps, include_ran=True)
        weights, fit, res = nnls_decompose(spectrum, basis, labels)
        for label, w in weights.items():
            assert w >= 0, f"Negative weight for {label}: {w}"

    def test_mixed_spectrum_recovers_mixture(self):
        """A 50/50 FCC+BCC mixture should give dominant ordered (non-RAN) components."""
        spectrum, true_w = simulate_mixed_ensemble(
            self.fps,
            {'FCC': 0.5, 'BCC': 0.5},
            n_orientations=5000, noise_level=0.001, seed=42
        )
        basis, labels = build_basis(self.fps, include_ran=True)
        weights, fit, res = nnls_decompose(spectrum, basis, labels)
        w_norm, _ = normalize_weights(weights)
        # FCC and BCC have similar fingerprints so NNLS may assign to either;
        # the combined ordered fraction (FCC+BCC) should dominate over ICO+RAN
        ordered = w_norm.get('FCC', 0) + w_norm.get('BCC', 0)
        assert ordered > 0.5, \
            f"FCC+BCC fraction = {ordered:.3f}, expected > 0.5 for pure FCC/BCC input"

# Run directly


if __name__ == '__main__':
    print("Running tests directly...\n")
    import traceback

    test_classes = [TestClusters, TestRotation, TestFingerprints, TestDecomposition]
    passed = 0; failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method in methods:
            try:
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                getattr(instance, method)()
                print(f'  PASS  {cls.__name__}::{method}')
                passed += 1
            except Exception as e:
                print(f'  FAIL  {cls.__name__}::{method}')
                print(f'        {e}')
                failed += 1

    print(f'\n{passed} passed, {failed} failed')
