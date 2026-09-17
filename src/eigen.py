"""
Eigenvalue algorithms from the course's linear-algebra unit (Chapra & Canale
ch. 27): the power method and the QR algorithm, written out rather than called
from NumPy, and used in Experiment 5.

Experiment 2 needed the spectrum of J* = -alpha ||r||_{1/alpha} C, where
C = diag(p*) - p* p*^T, and obtained it from `numpy.linalg.eigvalsh`.
Experiment 5 needs the same eigenvalues mu_j of C, because with a moving-average
estimator each of them becomes the stiffness of a damped oscillator
(derivation_exp5.md, Eq. 5.15). C is symmetric positive semidefinite, which is
exactly the case both classical algorithms are guaranteed on:

  power_method            largest |eigenvalue| and its eigenvector
  deflated_power_method   the full symmetric spectrum by Hotelling deflation
  qr_algorithm            the full symmetric spectrum by unshifted QR iteration
  gram_schmidt_qr         the QR factorization the iteration is built on

All three are validated against `numpy.linalg.eigvalsh` in the experiment and in
tests/test_estimators.py.
"""

from __future__ import annotations

import numpy as np


def power_method(A, x0=None, tol=1e-13, max_iter=10000):
    """
    Power method for the dominant eigenpair of a square matrix.

    x_{k+1} = A x_k / ||A x_k||, with the eigenvalue taken as the Rayleigh
    quotient x^T A x / x^T x (which converges twice as fast as the component
    ratio for a symmetric A). Converges when the dominant eigenvalue is unique
    in modulus, at the rate |lambda_2 / lambda_1|^k.

    Returns (eigenvalue, eigenvector, n_iterations, converged).
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("power_method: A must be square")
    x = np.ones(n) / np.sqrt(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    norm = np.linalg.norm(x)
    if norm == 0.0:
        raise ValueError("power_method: zero starting vector")
    x /= norm
    scale = max(1.0, float(np.max(np.abs(A))))
    lam = float(x @ A @ x)
    for it in range(1, max_iter + 1):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny == 0.0:                       # x already spans a null direction
            return 0.0, x, it, True
        x = y / ny
        lam = float(x @ A @ x)
        # stop on the residual, not on the change in lambda: for a symmetric A
        # the Rayleigh quotient is accurate to the SQUARE of the eigenvector
        # error, so a converged eigenvalue does not mean a converged vector --
        # and Hotelling deflation needs the vector.
        if float(np.linalg.norm(A @ x - lam * x)) <= tol * scale:
            return lam, x, it, True
    return lam, x, max_iter, False


def deflated_power_method(A, tol=1e-13, max_iter=20000):
    """
    Full spectrum of a SYMMETRIC matrix by Hotelling deflation: find the
    dominant eigenpair, subtract lambda v v^T, repeat. Returns eigenvalues in
    descending order of magnitude, and the matching eigenvectors as columns.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    if not np.allclose(A, A.T, atol=1e-12, rtol=0.0):
        raise ValueError("deflated_power_method: A must be symmetric")
    scale = float(np.max(np.abs(A)))
    vals = np.empty(n)
    vecs = np.empty((n, n))
    B = A
    rng = np.random.default_rng(0)
    for j in range(n):
        # start orthogonal to what has already been found, so the iteration
        # cannot drift back into an exhausted direction
        x0 = np.ones(n) / np.sqrt(n) if j == 0 else rng.normal(size=n)
        for i in range(j):
            x0 = x0 - (vecs[:, i] @ x0) * vecs[:, i]
        norm0 = np.linalg.norm(x0)
        if norm0 <= 1e-12:
            x0 = rng.normal(size=n)
            for i in range(j):
                x0 = x0 - (vecs[:, i] @ x0) * vecs[:, i]
            norm0 = np.linalg.norm(x0)
        x0 /= norm0
        if scale == 0.0 or float(np.max(np.abs(B))) <= tol * max(scale, 1.0):
            # the deflated remainder is numerically zero: every remaining
            # direction is a null vector, so keep the orthogonal start
            lam, v = 0.0, x0
        else:
            lam, v, _, _ = power_method(B, x0=x0, tol=tol, max_iter=max_iter)
            for i in range(j):                       # re-orthogonalize
                v = v - (vecs[:, i] @ v) * vecs[:, i]
            v = v / np.linalg.norm(v)
        vals[j] = lam
        vecs[:, j] = v
        B = B - lam * np.outer(v, v)
    return vals, vecs


def _fill_orthogonal_column(Q, j, m):
    """A unit vector orthogonal to the first j columns of Q. Needed when the
    j-th column of A is (numerically) a combination of the earlier ones: the
    Gram-Schmidt residual is then zero and leaving Q's column zero would
    destroy orthogonality, and with it the similarity RQ = Q^T A Q that the QR
    algorithm relies on."""
    best, best_norm = None, 0.0
    for k in range(m):
        v = np.zeros(m)
        v[k] = 1.0
        for _ in range(2):
            for i in range(j):
                v -= (Q[:, i] @ v) * Q[:, i]
        nv = np.linalg.norm(v)
        if nv > best_norm:
            best, best_norm = v, nv
    # the complement of j < m orthonormal vectors always contains some e_k with
    # residual norm >= sqrt(1 - j/m) > 0; anything tiny means Q was not orthonormal
    if best is None or best_norm < 1e-8:
        raise np.linalg.LinAlgError("could not extend Q to an orthonormal basis")
    return best / best_norm


def gram_schmidt_qr(A, tol=1e-14):
    """
    QR factorization by modified Gram-Schmidt with one re-orthogonalization
    pass (numerically far better behaved than the classical version), and a
    rank-deficiency guard.

    Returns (Q, R) with Q orthonormal and R upper triangular. When column j of
    A lies in the span of the earlier columns, R[j, j] is set to zero and Q's
    column is completed to an orthonormal basis, so Q stays orthogonal and
    QR = A still holds. This matters here because the softmax covariance
    diag(p) - p p^T is singular by construction.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        col_norm = np.linalg.norm(A[:, j])
        v = A[:, j].copy()
        for _ in range(2):                # modified Gram-Schmidt, twice
            for i in range(j):
                c = Q[:, i] @ v
                R[i, j] += c
                v = v - c * Q[:, i]
        nv = np.linalg.norm(v)
        if nv <= tol * max(1.0, col_norm):
            R[j, j] = 0.0
            Q[:, j] = _fill_orthogonal_column(Q, j, m)
        else:
            R[j, j] = nv
            Q[:, j] = v / nv
    return Q, R


def qr_algorithm(A, tol=1e-13, max_iter=20000):
    """
    Unshifted QR algorithm for a SYMMETRIC matrix: A_0 = A, A_{k+1} = R_k Q_k
    where A_k = Q_k R_k. The iterates stay similar to A and converge to a
    diagonal matrix of eigenvalues, at the rate |lambda_{i+1}/lambda_i|^k.

    Returns (eigenvalues ascending, n_iterations, off_diagonal_norm).
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("qr_algorithm: A must be square")
    if not np.allclose(A, A.T, atol=1e-12, rtol=0.0):
        raise ValueError("qr_algorithm: this unshifted implementation assumes symmetry")
    scale = max(1.0, float(np.max(np.abs(A))))
    for it in range(1, max_iter + 1):
        off = float(np.sqrt(np.sum(A ** 2) - np.sum(np.diag(A) ** 2)))
        if off <= tol * scale:
            break
        Q, R = gram_schmidt_qr(A)
        A = R @ Q
    off = float(np.sqrt(max(np.sum(A ** 2) - np.sum(np.diag(A) ** 2), 0.0)))
    return np.sort(np.diag(A).copy()), it, off
