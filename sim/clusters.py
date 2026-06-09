"""
Unit cluster coordinates for archetypal short-range order polyhedra.

Each function returns an (N, 3) array of 3D Cartesian coordinates
for the cluster, centered at the origin with nearest-neighbor
distance normalized to 1.

Clusters implemented:
    FCC  - 13-atom face-centered cubic (central + 12 nearest neighbors)
    BCC  - 9-atom body-centered cubic  (central + 8 nearest neighbors)
    HCP  - 13-atom hexagonal close-packed
    ICO  - 13-atom icosahedral
    SC   - 7-atom simple cubic (central + 6 nearest neighbors)

Reference: Liu et al., Phys. Rev. Lett. 116, 205501 (2016)
"""

import numpy as np


def fcc_cluster():
    """
    13-atom FCC cluster: central atom + 12 nearest neighbors.
    Nearest-neighbor distance = 1.
    """
    # FCC nearest neighbors are at a/sqrt(2) along <110> directions
    # We set this distance to 1, so a = sqrt(2)
    a = np.sqrt(2.0)
    neighbors = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            neighbors.append([i * a / 2, j * a / 2, 0.0])
            neighbors.append([i * a / 2, 0.0, j * a / 2])
            neighbors.append([0.0, i * a / 2, j * a / 2])
    coords = np.array([[0.0, 0.0, 0.0]] + neighbors)
    # normalize so nn distance = 1
    nn_dist = np.linalg.norm(coords[1])
    return coords / nn_dist


def bcc_cluster():
    """
    9-atom BCC cluster: central atom + 8 nearest neighbors.
    Nearest-neighbor distance = 1.
    """
    # BCC nn along <111>; distance = a*sqrt(3)/2, set to 1 => a = 2/sqrt(3)
    a = 2.0 / np.sqrt(3.0)
    neighbors = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                neighbors.append([i * a / 2, j * a / 2, k * a / 2])
    coords = np.array([[0.0, 0.0, 0.0]] + neighbors)
    nn_dist = np.linalg.norm(coords[1])
    return coords / nn_dist


def hcp_cluster():
    """
    13-atom HCP cluster: central atom + 12 nearest neighbors
    (6 in-plane + 3 above + 3 below).
    Nearest-neighbor distance = 1.
    """
    # 6 in-plane neighbors at angles 0, 60, 120, 180, 240, 300 degrees
    in_plane = []
    for k in range(6):
        angle = k * np.pi / 3.0
        in_plane.append([np.cos(angle), np.sin(angle), 0.0])

    # c/a ratio for ideal HCP: c/a = sqrt(8/3)
    ca = np.sqrt(8.0 / 3.0)
    # 3 neighbors above (offset by a/2 in x, a/(2sqrt(3)) in y, c/2 in z)
    above = []
    below = []
    for k in range(3):
        angle = np.pi / 6.0 + k * 2.0 * np.pi / 3.0
        x = np.cos(angle) / np.sqrt(3.0)
        y = np.sin(angle) / np.sqrt(3.0)
        above.append([x, y, ca / 2.0])
        below.append([x, y, -ca / 2.0])

    neighbors = in_plane + above + below
    coords = np.array([[0.0, 0.0, 0.0]] + neighbors)
    nn_dist = np.linalg.norm(coords[1])
    return coords / nn_dist


def ico_cluster():
    """
    13-atom icosahedral cluster: central atom + 12 vertices.
    Nearest-neighbor distance = 1.
    Uses the golden-ratio construction.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    # 12 vertices of icosahedron from permutations of (0, ±1, ±phi)
    vertices = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            vertices.append([0.0, s1 * 1.0, s2 * phi])
            vertices.append([s1 * 1.0, s2 * phi, 0.0])
            vertices.append([s2 * phi, 0.0, s1 * 1.0])
    coords = np.array([[0.0, 0.0, 0.0]] + vertices)
    nn_dist = np.linalg.norm(coords[1])
    return coords / nn_dist


def sc_cluster():
    """
    7-atom simple cubic cluster: central atom + 6 nearest neighbors
    along ±x, ±y, ±z axes.
    Nearest-neighbor distance = 1.
    """
    neighbors = [
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ]
    coords = np.array([[0.0, 0.0, 0.0]] + neighbors)
    return coords


def get_all_clusters():
    """Return dict of all cluster coordinate arrays."""
    return {
        'FCC': fcc_cluster(),
        'BCC': bcc_cluster(),
        'HCP': hcp_cluster(),
        'ICO': ico_cluster(),
        'SC':  sc_cluster(),
    }


def validate_clusters():
    """Print nn distances and atom counts for sanity check."""
    clusters = get_all_clusters()
    print(f"{'Cluster':6s}  {'N_atoms':8s}  {'nn_dist':10s}  {'expected':10s}")
    print("-" * 44)
    for name, coords in clusters.items():
        # distance from center to all others
        dists = np.linalg.norm(coords[1:], axis=1)
        nn = dists.min()
        print(f"{name:6s}  {len(coords):8d}  {nn:10.6f}  {'1.000000':10s}")


if __name__ == "__main__":
    validate_clusters()
