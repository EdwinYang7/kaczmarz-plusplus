import numpy as np
from typing import Tuple
from scipy.optimize import root_scalar
from scipy.linalg import hadamard
from numpy.linalg import pinv

from tqdm import tqdm
import random

from sketch import Sketch, SketchFactory, GaussianSketchFactory
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import cg, lsqr, LinearOperator
from utils import fht


def bernoulli_expo(t, nu):
    p = 2**(-t / nu)
    return True if random.random() < p else False


def bernoulli_frac(t, nu):
    p = 1 / (1 + t / nu)
    return True if random.random() < p else False


def get_accelerated_params(mu: float, nu: float) -> Tuple[float, float, float]:
    beta = 1 - np.sqrt(mu / nu)
    gamma = np.sqrt(1 / (mu * nu))
    alpha = 1 / (1 + gamma * nu)

    return beta, gamma, alpha


def inner_precondition(
    A_S: np.ndarray,
    tau: int,
    lam: float,
    seed: int=None
):
    k, n = A_S.shape
    rng = np.random.default_rng(seed)
    ### RHT on right hand side
    D = rng.choice([-1, 1], size=n)
    B = A_S * D
    H = hadamard(n) / np.sqrt(n)
    C = B @ H
    ### Subsampled RHT
    idx = rng.choice(n, size=tau, replace=False)
    scale = np.sqrt(n / tau)
    A_hat = scale * C[:,idx]
    G = A_hat @ A_hat.T + lam * np.eye(k)
    R = cholesky(G, lower=False)
    return R


def inner_solve(
    A_S: np.ndarray,
    tau: int,
    lam: float,
    r_t: np.ndarray,
    seed: int = None,
    atol: float = 1e-8,
    btol: float = 1e-8,
    iter_lim: int = None
):
    k, n = A_S.shape
    R = inner_precondition(A_S, tau, lam, seed)
    B = np.hstack([A_S, np.sqrt(lam) * np.eye(k)])
    M = solve_triangular(R.T, B, lower=True)
    b = solve_triangular(R.T, r_t, lower=True)
    x = lsqr(M, b, atol=atol, btol=btol, iter_lim=iter_lim)[0]
    w_t = x[:n]
    v_t = x[n:]
    return w_t, v_t


def kaczmarz_plusplus(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    accelerated=False,
    block=True,
    exact=False,
    reg=1e-6,
    acc=1e-5,
    maxiter=20,
    rng=None,
    accuracy=0
):
    m, n = A.shape
    k = Sf.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    z_param = 1.0 if accelerated else 0.0
    eta = 0.0
    nu = 2*n/k

    X = np.zeros((1 + t_max, n))
    x = X[0, :] = x0.copy()
    v = x.copy()

    z = np.zeros(n)
    skip = round(n/k+1)
    cnt = 1.0
    ratio = 0.0
    S_list = []
    cholesky_list = []
    dist_new = dist_old = 0.0

    for t in tqdm(range(1, t_max + 1)):
        if not block:
            S = Sf()
            SA = S @ A
            G = SA @ SA.T
            L = cholesky(G + reg * np.eye(k), lower=True)

        elif random.random() < min(1,n*np.log(n)/(k*t)):
            S = Sf()
            S_list.append(S)
            SA = S @ A
            G = SA @ SA.T

            L = cholesky(G + reg * np.eye(k), lower=True)
            cholesky_list.append(L)
        else:
            idx = random.randint(0, len(cholesky_list) - 1)
            S = S_list[idx]
            SA = S @ A
            L = cholesky_list[idx]

        b_ = SA @ x - S @ b
        if exact:
            u_ = solve_triangular(L, b_, lower=True)
            u = solve_triangular(L.T, u_, lower=False)
            w = SA.T @ u
        else:
            tau = 2*k
            w, v = inner_solve(SA, tau, lam=reg, r_t=b_, seed=42, iter_lim=maxiter)
            
        z = z_param * (z + w)
        x = x - w - eta * z
        X[t, :] = x.copy()

        if t % (2*skip) <= skip:
            dist_old += np.linalg.norm(b_)**2
        else:
            dist_new += np.linalg.norm(b_)**2

        if np.linalg.norm(A @ x - b) / np.linalg.norm(b) <= accuracy:
            print(f"Converged at iteration {t} with dist_new = {np.sqrt(dist_new)}")
            break

        if accelerated and t % (2*skip) == 0:
            a_old = cnt**np.log(cnt)
            a_new = (cnt+1)**np.log(cnt+1)
            ratio = ratio*(a_old/a_new) + min(1,dist_new / dist_old) * (1 - a_old/a_new)
            rho = max(0,1 - ratio**(1/skip))
            z_param = (1-rho)/(1+rho)
            eta = 1/nu
            cnt = cnt + 1
            dist_new = dist_old = 0.0

    return X


def coordinate_descent_meta(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    accelerated=True,
    block=True,
    beta=None,
    gamma=None,
    alpha=None,
    reg=0,   # Inner regularization
    rng=None,
    accuracy=0,
):
    m, n = A.shape
    k = Sf.shape[0]
    flops_cd = [0]

    if rng is None:
        rng = np.random.default_rng()

    z_param = 1.0 if accelerated else 0.0
    eta = 0.0
    nu = 2*n/k

    X = np.zeros((1 + t_max, n))
    x = X[0, :] = x0.copy()
    z = np.zeros(n)
    skip = round(n/k+1)
    cnt = 1.0
    ratio = 0.0
    S_list = []
    cholesky_list = []
    dist_new = dist_old = 0.0
    for t in tqdm(range(1, t_max + 1)):
        flops = flops_cd[t-1]
        if not block:
            S = Sf()
            SA = S @ A
            SAS = S @ SA.T
            L = cholesky(SAS + reg * np.eye(k), lower=True)
            flops += k**3 / 3
        elif random.random() < min(1,n*np.log(n)/(k*t)): # t < 2 or bernoulli_frac(t, 5*nu): #    # Sample new block and compute block inverse
            S = Sf()
            S_list.append(S)
            SA = S @ A
            SAS = S @ SA.T
            # SAS_inv = pinv(SAS + reg * np.eye(k))
            L = cholesky(SAS + reg * np.eye(k), lower=True)
            flops += k**3 / 3
            cholesky_list.append(L)
        else:
            idx = random.randint(0, len(cholesky_list) - 1)
            S = S_list[idx]
            SA = S @ A
            L = cholesky_list[idx]

        b_ = SA @ x - S @ b
        flops += 2 * n * k
        u_ = solve_triangular(L, b_, lower=True)
        u = solve_triangular(L.T, u_, lower=False)
        flops += 2 * k**2
        w = S.T @ u
        z = z_param * (z + w)
        x = x - w - eta * z
        flops = flops + 2 * (k + n) if accelerated else flops + k
        X[t, :] = x.copy()

        if t % (2*skip) <= skip:
            dist_old += np.linalg.norm(b_)**2
        else:
            dist_new += np.linalg.norm(b_)**2
        flops += 2*k-1

        # if np.sqrt(dist_new) / np.linalg.norm(b) <= accuracy:
        if np.linalg.norm(A @ x - b) / np.linalg.norm(b) <= accuracy:
            print(f"Converged at iteration {t} with dist_new = {np.sqrt(dist_new)}")
            break

        if accelerated and t % (2*skip) == 0:
            a_old = cnt**np.log(cnt)
            a_new = (cnt+1)**np.log(cnt+1)
            ratio = ratio*(a_old/a_new) + min(1,dist_new / dist_old) * (1 - a_old/a_new)
            rho = max(0,1 - ratio**(1/skip))
            z_param = (1-rho)/(1+rho)
            eta = 1/nu
            # eta = max(0, (1/nu - rho)/(1-rho))
            cnt = cnt + 1
            dist_new = dist_old = 0.0

        flops_cd.append(flops)

    return X, np.array(flops_cd)