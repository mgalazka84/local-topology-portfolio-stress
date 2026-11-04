"""Restricted QP/SOCP master in Clarabel, using its native Python interface."""
import numpy as np
import scipy.sparse as sp
import clarabel


def solve_master_conic(matrices, previous, cost, start=None, ftol=1e-11):
    H = np.asarray(matrices)
    n = len(previous)
    c = np.broadcast_to(np.asarray(cost, dtype=float), n)
    single = len(H) == 1
    dim = 2*n if single else 2*n+1
    P = np.zeros((dim, dim))
    q = np.zeros(dim)
    q[n:2*n] = c
    if single:
        P[:n, :n] = 2*H[0]
    else:
        q[-1] = 1
    A = np.zeros((1+3*n, dim))
    A[0, :n] = 1
    A[1:1+n, :n] = -np.eye(n)
    A[1+n:1+2*n, :n] = np.eye(n)
    A[1+n:1+2*n, n:2*n] = -np.eye(n)
    A[1+2*n:, :n] = -np.eye(n)
    A[1+2*n:, n:2*n] = -np.eye(n)
    bs = [np.r_[1., np.zeros(n), previous, -previous]]
    blocks = [sp.csc_matrix(A)]
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(3*n)]
    if not single:
        for matrix in H:
            B = np.linalg.cholesky(matrix).T
            C = np.zeros((n+2, dim))
            C[0, -1] = C[-1, -1] = -1
            C[1:-1, :n] = -2*B
            blocks.append(sp.csc_matrix(C))
            bs.append(np.r_[1., np.zeros(n), -1.])
            cones.append(clarabel.SecondOrderConeT(n+2))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.max_threads = 1
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = ftol
    solver = clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P)).tocsc(), q,
        sp.vstack(blocks).tocsc(), np.concatenate(bs), cones, settings)
    result = solver.solve()
    x = np.asarray(result.x)
    if not np.all(np.isfinite(x)) or np.maximum(x[:n], 0).sum() == 0:
        raise RuntimeError('No finite feasible candidate from Clarabel: '+str(result.status))
    w = np.maximum(x[:n], 0)
    w /= w.sum()
    status = str(result.status)
    # Restore the simple dual domains, then evaluate a quadratic Lagrangian
    # lower bound independently of Clarabel's reported objective/gap.
    dual = np.asarray(result.z)
    if single:
        alpha = np.ones(1)
    else:
        soc = dual[1+3*n:].reshape(len(H), n+2)
        alpha = np.maximum(soc[:, 0]+soc[:, -1], 0)
        alpha /= alpha.sum()
    h = np.clip(dual[1+n:1+2*n]-dual[1+2*n:1+3*n], -c, c)
    z = np.maximum(dual[1:1+n], 0)
    eta = -dual[0]
    mixed = np.einsum('k,kij->ij', alpha, H)
    b = h-eta-z
    quadratic_lower = eta-h@previous-.25*b@np.linalg.solve(mixed, b)
    return w, {'success': status == 'Solved', 'message': status,
        'iterations': int(result.iterations),
        'constraint_residual': float(result.r_prim),
        'dual_residual': float(result.r_dual),
        'primal_dual_gap': float(result.obj_val-result.obj_val_dual),
        'quadratic_lower': float(quadratic_lower)}
