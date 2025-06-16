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


def calculate_lamda_mu_nu(A: np.ndarray, k: int) -> Tuple[float, float, float]:

    s2 = np.linalg.eigvalsh(A)  # (A.T @ A)
    s2[s2 < np.mean(s2) * 1e-8] = 0
    def f(lamda):
        return np.sum(s2 / (s2 + lamda)) - k

    lamda = root_scalar(f, bracket=[1e-8, np.max(s2) * len(s2) * 1e8]).root
    s2_min = np.min(s2[s2 > 0])
    mu = s2_min / (s2_min + lamda)
    num = np.sum((s2 > 0) / (s2 + lamda))
    denom = np.sum(s2 / (s2 + lamda) ** 2)
    lamda_prime = lamda * num / denom
    nu = 1 + lamda_prime / (s2_min + lamda)

    return lamda, mu, nu


def estimate_lamda_nu(evs: np.ndarray, k: int) -> Tuple[float, float]:
    """Estimate the parameters lamda, nu for the accelerated Kaczmarz method using the sketched matrix.

    Parameters
    ----------
    evs : np.ndarray
        The eigenvalues of the sketched matrix S @ A @ S.T.
    k : int
        The number of rows in the sketched matrix.

    Returns
    -------
    Tuple[float, float]
        The estimated values of lamda, nu.

    """

    p = evs.size
    alpha = k / p

    pinv_evs = np.zeros_like(evs)
    pinv_evs[evs > 0] = 1 / evs[evs > 0]

    neg_f_prime = np.mean(pinv_evs)

    lamda = alpha / neg_f_prime

    f_prime2 = np.mean(pinv_evs**2)
    lamda_prime = 2 * neg_f_prime**2 / lamda / f_prime2

    return lamda, lamda_prime


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


def kaczmarz_plus_plus(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    accelerated=False,
    exact=False,
    beta=None,
    gamma=None,
    alpha=None,
    reg=1e-6,
    acc=1e-5,
    maxiter=20,
    rng=None,
):
    m, n = A.shape
    k = Sf.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if accelerated:
        if None in (beta, gamma, alpha):
            lamda, mu, nu = calculate_lamda_mu_nu(A.T @ A, k)
            beta, gamma, alpha = get_accelerated_params(mu, nu)
    else:
        beta, gamma, alpha = 1.0, 1.0, 0.0

    X = np.zeros((1 + t_max, n))
    x = X[0, :] = x0.copy()
    v = x.copy()

    for t in tqdm(range(1, t_max + 1)):
        y = alpha * v + (1 - alpha) * x
        S = Sf()
        SA = S @ A
        r = SA @ y - S @ b
        G = SA @ SA.T
        if reg > 0:
            G = G + reg * np.eye(k)

        if exact:
            L = cholesky(G, lower=True)
            u_ = solve_triangular(L, r, lower=True)
            u  = solve_triangular(L.T, u_, lower=False)

        else:
            # u = np.linalg.solve(G, r)
            u, info = cg(G, r, rtol=acc, maxiter=maxiter)

        w = SA.T @ u
        x = y - w
        v = beta * v + (1 - beta) * y - gamma * w

        X[t, :] = x.copy()

    return X


def randomized_kaczmarz(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    accelerated=False,
    beta=None,
    gamma=None,
    alpha=None,
    rng=None,
):
    m, n = A.shape
    k = Sf.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if accelerated:
        if None in [beta, gamma, alpha]:
            lamda, mu, nu = calculate_lamda_mu_nu(A.T @ A, k)
            beta, gamma, alpha = get_accelerated_params(mu, nu)
    else:
        beta = 1.0
        gamma = 1.0
        alpha = 0.0

    X = np.zeros((1 + t_max, n))
    x = X[0, :] = x0.copy()
    v = x

    for t in tqdm(range(1, t_max + 1)):
        y = alpha * v + (1 - alpha) * x
        S = Sf()
        SA = S @ A
        b_ = SA @ y - S @ b
        u = np.linalg.solve(SA @ SA.T, b_)
        w = SA.T @ u
        x = y - w
        v = beta * v + (1 - beta) * y - gamma * w
        X[t, :] = x.copy()

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


def coordinate_descent(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    accelerated=False,
    beta=None,
    gamma=None,
    alpha=None,
    rng=None,
):
    m, n = A.shape
    k = Sf.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if accelerated:
        if None in [beta, gamma, alpha]:
            lamda, mu, nu = calculate_lamda_mu_nu(A, k)
            beta, gamma, alpha = get_accelerated_params(mu, nu)
    else:
        beta = 1.0
        gamma = 1.0
        alpha = 0.0

    X = np.zeros((1 + t_max, n))
    x = X[0, :] = x0.copy()
    v = x

    for t in tqdm(range(1, t_max + 1)):
        y = alpha * v + (1 - alpha) * x
        S = Sf()
        SA = S @ A
        SAS = S @ SA.T
        b_ = SA @ y - S @ b
        u = np.linalg.solve(SAS, b_)
        w = S.T @ u
        x = y - w
        v = beta * v + (1 - beta) * y - gamma * w
        X[t, :] = x.copy()

    return X


### Update acceleration parameters based on A norm
def coordinate_descent_tuned_A(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    reg=0,
    rng=None,
):
    m, n = A.shape
    k = Sf.shape[0]
    flops_cd_heuristic = [0]

    if rng is None:
        rng = np.random.default_rng()

    z_param = 1.0
    eta = 0.0
    nu = 2*n/k

    X = np.zeros((1 + t_max, n))
    x = X[0, :] = x0.copy()
    z = np.zeros(n)
    skip = round(n/k+1)
    cnt = 1.0
    ratio = 0.0
    
    for t in tqdm(range(1, t_max + 1)):
        flops = flops_cd_heuristic[t-1]
        S = Sf()   # assume uniform sampling, does not have FLOPs
        SA = S @ A
        SAS = S @ SA.T
        b_ = SA @ x - S @ b
        flops += 2 * n * k
        # u = np.linalg.solve(SAS + reg * np.eye(k), b_)
        L = cholesky(SAS + reg * np.eye(k), lower=True)
        u_ = solve_triangular(L, b_, lower=True)
        u = solve_triangular(L.T, u_, lower=False)
        flops += (k**3) / 3 + 2 * k**2
        w = S.T @ u
        z = z_param * (z + w) 
        flops += k + n
        x = x - w - eta * z
        flops += k + n
        X[t, :] = x.copy() 
        if t % (2*skip) == 0:
            dist_new = ((x - X[t-skip,:]).T @ A @ (x - X[t-skip,:]))
            flops += 2 * n**2 + 2 * n - 1
            dist_old = ((X[t-skip,:] - X[t-2*skip,:]).T @ A @ (X[t-skip,:] - X[t-2*skip,:]))
            flops += 2 * n**2 + 2 * n - 1
            a_old = cnt**np.log(cnt)
            a_new = (cnt+1)**np.log(cnt+1)
            ratio = ratio*(a_old/a_new) + min(1,dist_new / dist_old) * (1 - a_old/a_new)
            rho = max(0,1 - ratio**(1/skip))
            #print("t=%d, skip=%d:   rho = %f\n" % (t,skip,rho))
            z_param = (1-rho)/(1+rho)
            eta = max(0, (1/nu - rho)/(1-rho))
            cnt = cnt + 1
        flops_cd_heuristic.append(flops)
    return X, np.array(flops_cd_heuristic)


### Update acceleration parameters based on l2 norm
def coordinate_descent_tuned_l2(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    reg=0,
    rng=None,
):
    m, n = A.shape
    k = Sf.shape[0]
    flops_cd_heuristic = [0]

    if rng is None:
        rng = np.random.default_rng()

    z_param = 1.0
    eta = 0.0
    nu = 2*n/k

    X = np.zeros((1 + t_max, n))
    x = X[0, :] = x0.copy()
    z = np.zeros(n)
    skip = round(n/k+1)
    cnt = 1.0
    ratio = 0.0
    dist_new = dist_old = 0.0
    for t in tqdm(range(1, t_max + 1)):
        flops = flops_cd_heuristic[t-1]
        S = Sf()   # assume uniform sampling, does not have FLOPs
        SA = S @ A
        SAS = S @ SA.T
        b_ = SA @ x - S @ b
        flops += 2 * n * k
        # u = np.linalg.solve(SAS + reg * np.eye(k), b_)
        L = cholesky(SAS + reg * np.eye(k), lower=True)
        u_ = solve_triangular(L, b_, lower=True)
        u = solve_triangular(L.T, u_, lower=False)
        flops += (k**3) / 3 + 2 * k**2
        w = S.T @ u
        z = z_param * (z + w) 
        flops += k + n
        x = x - w - eta * z
        flops += k + n
        X[t, :] = x.copy()

        if t % (2*skip) <= skip:
            dist_old += np.linalg.norm(b_)**2
        else:
            dist_new += np.linalg.norm(b_)**2
        flops += 2*k-1
        
        if t % (2*skip) == 0:
            a_old = cnt**np.log(cnt)
            a_new = (cnt+1)**np.log(cnt+1)
            ratio = ratio*(a_old/a_new) + min(1,dist_new / dist_old) * (1 - a_old/a_new)
            rho = max(0,1 - ratio**(1/skip))
            #print("t=%d, skip=%d:   rho = %f\n" % (t, skip, rho))
            z_param = (1-rho)/(1+rho)
            eta = max(0, (1/nu - rho)/(1-rho))
            cnt = cnt + 1
            dist_new = dist_old = 0.0
        flops_cd_heuristic.append(flops)
    return X, np.array(flops_cd_heuristic)


def coordinate_descent_block(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Sf: SketchFactory,
    t_max: int,
    accelerated=True,
    reg=0,
    rng=None,
    accuracy=0,
):
    m, n = A.shape
    k = Sf.shape[0]
    flops_cd_block = [0]

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
    error_ratio_list = []
    dist_new = dist_old = 0.0
    for t in tqdm(range(1, t_max + 1)):
        flops = flops_cd_block[t-1]
        if  random.random() < min(1,n*np.log(n)/(k*t)): # t < 2 or bernoulli_frac(t, 5*nu): #    # Sample new block and compute block inverse
            S = Sf()
            S_list.append(S)
            SA = S @ A
            SAS = S @ SA.T   # SAS is positive definite
            # SAS_inv = pinv(SAS + reg * np.eye(k))
            L = cholesky(SAS + reg * np.eye(k), lower=True)
            flops += k**3 / 3
            cholesky_list.append(L)
        else:   # Uniformly sample a previously sampled block inverse
            idx = random.randint(0, len(cholesky_list) - 1)
            S = S_list[idx]
            SA = S @ A
            L = cholesky_list[idx]

        b_ = SA @ x - S @ b
        flops += 2 * n * k
        u_ = solve_triangular(L, b_, lower=True)
        u = solve_triangular(L.T, u_, lower=False)
        flops += 2 * k**2
        # u = SAS_inv @ b_
        # flops += 2 * k**2 - k
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

        if accelerated and t % (2*skip) == 0:
            # if np.sqrt(dist_new) / np.linalg.norm(b) <= accuracy:
            if np.linalg.norm(A @ x - b) / np.linalg.norm(b) <= accuracy:
                print(f"Converged at iteration {t} with dist_new = {np.sqrt(dist_new)}")
                break
            a_old = cnt**np.log(cnt)
            a_new = (cnt+1)**np.log(cnt+1)
            ratio = ratio*(a_old/a_new) + min(1,dist_new / dist_old) * (1 - a_old/a_new)
            rho = max(0,1 - ratio**(1/skip))
            z_param = (1-rho)/(1+rho)
            eta = 1/nu
            # eta = max(0, (1/nu - rho)/(1-rho))
            cnt = cnt + 1
            dist_new = dist_old = 0.0            
        flops_cd_block.append(flops)

        # error_ratio = np.sqrt(dist_new) / np.linalg.norm(A @ x - b)
        # if error_ratio > 0:
        #     error_ratio_list.append(error_ratio)

    # print(f"Residual = {np.linalg.norm(A @ x - b)/np.linalg.norm(b)}, Estimate = {np.sqrt(dist_new)/np.linalg.norm(b)}")
    # print(f"Ratio of estimate error over real residual: min = {min(error_ratio_list)}, max = {max(error_ratio_list)}.")
        
    return X, np.array(flops_cd_block)


# def coordinate_descent_experimental(
#     A: np.ndarray,
#     b: np.ndarray,
#     x0: np.ndarray,
#     Sf: SketchFactory,
#     t_max: int,
#     x_star=None,
#     accelerated=False,
#     tuned=True,
#     beta=None,
#     gamma=None,
#     alpha=None,
#     rng=None,
# ):
#     m, n = A.shape
#     k = Sf.shape[0]

#     if rng is None:
#         rng = np.random.default_rng()

#     z_param = 1.0
#     w_param = 0.0
#     eta = 0.0

#     if accelerated and (not tuned or x_star is not None):
#         if None in [beta, gamma, alpha]:
#             lamda, mu, nu = calculate_lamda_mu_nu(A, k)
#             beta, gamma, alpha = get_accelerated_params(mu, nu)
#             if not tuned:
#                 z_param = (1-alpha)*beta
#                 w_param = alpha*(gamma-1)
#                 eta = w_param / z_param
#                 print("\nmu="+str(mu)+" nu="+str(nu)+" n/k="+str(n/k))
#                 print("beta="+str(beta)+" gamma="+str(gamma)+" alpha="+str(alpha)+" rho="+str(1-beta))
#                 print("z <- "+str(z_param)+"*z + "+str(w_param)+"*w"+ " (tuned = "+str(tuned)+")")
# #                rho = 1-beta
# #                nu = min(2*n/k, 1/rho)
# #                z_param = (1-rho)/(1+rho)
# #                w_param = (1/nu - rho)/(1+rho)
#     else:
#         beta = 1.0
#         gamma = 1.0
#         alpha = 0.0
#         nu = 2*n/k

#     X = np.zeros((1 + t_max, n))
#     y = X[0, :] = x0.copy()
#     z = np.zeros(n)
#     skip = round(n/k+1)
#     cnt = 1.0
#     ratio = 0.0
    
#     for t in tqdm(range(1, t_max + 1)):        
#         S = Sf()
#         SA = S @ A
#         SAS = S @ SA.T
#         b_ = SA @ y - S @ b
#         u = np.linalg.solve(SAS, b_)  
#         w = S.T @ u
#         z = z_param * (z + w)  #z_param * z + w_param * w
#         y = y - w - eta * z
#         X[t, :] = y.copy() #(y-z).copy()
#         if accelerated and tuned and t % (2*skip) == 0:
#             if x_star is not None:
#                 dist_new = ((y - x_star).T @ A @ (y - x_star))
#                 dist_old = ((X[t-skip,:] - x_star).T @ A @ (X[t-skip,:] - x_star))
#                 a_old = 0
#                 a_new = 1
#             else:
#                 dist_new = ((y - X[t-skip,:]).T @ A @ (y - X[t-skip,:]))
#                 dist_old = ((X[t-skip,:] - X[t-2*skip,:]).T @ A @ (X[t-skip,:] - X[t-2*skip,:]))
#                 a_old = cnt**np.log(cnt)
#                 a_new = (cnt+1)**np.log(cnt+1)
#             ratio = ratio*(a_old/a_new) + min(1,dist_new / dist_old) * (1 - a_old/a_new)
#             rho = max(0,1 - ratio**(1/skip))
#             z_param = (1-rho)/(1+rho)
#             w_param = max(0, (1/nu - rho)/(1+rho))
#             eta = max(0, (1/nu - rho)/(1-rho))
#             #print("\n"+str(t)+": z <- "+str(z_param)+"*z + "+str(w_param)+"*w"+ "   (rho = "+str(rho)+", ratio="+str(ratio)+")")
#             cnt = cnt + 1
#     return X

