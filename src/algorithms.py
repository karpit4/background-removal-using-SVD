"""Algorithms for low-rank matrix approximation."""

import numpy as np


def trunc_svd(M, r):
    """
    Compute the truncated (exact) SVD of M up to rank r.

    Returns
    -------
    U : np.ndarray
        Left singular vectors, shape (m, r).
    S : np.ndarray
        Singular values, shape (r,).
    Vt : np.ndarray
        Transposed right singular vectors, shape (r, n).

    The low-rank approximation can be reconstructed as:
        M_r = U @ np.diag(S) @ Vt
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    return U[:, :r], S[:r], Vt[:r, :]


def rand_svd(M, r, oversampling=10):
    """
    Compute a randomized SVD approximation of M of rank r.

    Returns
    -------
    U : np.ndarray
        Approximate left singular vectors, shape (m, r).
    S : np.ndarray
        Approximate singular values, shape (r,).
    Vt : np.ndarray
        Approximate transposed right singular vectors, shape (r, n).

    The low-rank approximation can be reconstructed as:
        M_r = U @ np.diag(S) @ Vt
    """
    _, n = M.shape
    l = r + oversampling

    # Randomized Range Finder
    Omega = np.random.randn(n, l)
    Y = M @ Omega
    Q, _ = np.linalg.qr(Y)

    # SVD of the reduced matrix
    B = Q.T @ M
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Recover approximate left singular vectors
    U = Q @ U_tilde

    return U[:, :r], S[:r], Vt[:r, :]
