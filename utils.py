import numpy as np
from sketch import GaussianSketchFactory, SubsamplingSketchFactory



def hadamard(n):
    """
    Generate a Hadamard matrix of size n
    """
    if n == 1:
        return np.array([[1]])
    else:
        H_n_1 = hadamard(n // 2)
        top = np.hstack((H_n_1, H_n_1))
        bottom = np.hstack((H_n_1, -H_n_1))
        H_n = np.vstack((top, bottom))
        return H_n


def rht(n):
    """
    Generates the Randomized Hadamard Transform matrix Q of size n x n,
    working for any positive integer n.
    """
    N = 1 << (n - 1).bit_length()   # Find the smallest power of 2 larger than n
    H_N = hadamard(N)
    random_signs = np.random.choice([-1, 1], size=N)
    D_N = np.diag(random_signs)
    Q_N = (1 / np.sqrt(n)) * H_N @ D_N
    
    Q = Q_N[:n, :n]   # Extract the first n rows and n columns to get Q of size n x n
    
    return Q


# def fht(indices, A):
#     """
#     Fast Hadamard Transform with FLOPs counting.
#     Assume that n is power of 2.
#     """
#     flops = 0
#     n = A.shape[0]
#     if n == 1:
#         return A, flops
#     i1 = indices[indices < n // 2]
#     i2 = indices[indices >= n // 2]
#     if len(i1) == 0:
#         return fht(i2 - n//2, A[:n//2, :] - A[n//2:, :]), flops
#     elif len(i2) == 0:
#         return fht(i1, A[:n//2, :] + A[n//2:, :]), flops
#     else:
#         A1, flops1 = fht(i1, A[:n//2, :] + A[n//2:, :])
#         A2, flops2 = fht(i2 - n // 2, A[:n//2, :] - A[n//2:, :])
#         flops += flops1 + flops2 + n
#         A = np.concatenate([A1, A2], axis=0)
#         return A, flops
    

def fht(A):
    """
    Fast Hadamard Transform with FLOPs counting.
    Assume that n is power of 2.
    """
    flops = 0
    if A.ndim == 1:   # vector case
        A = A.reshape((-1,1))
    n, m = A.shape
    if n == 1:
        return A, flops
    A1, flops1 = fht(A[:n//2, :] + A[n//2:, :])
    A2, flops2 = fht(A[:n//2, :] - A[n//2:, :])
    flops += flops1 + flops2 + n*m
    A = np.concatenate([A1, A2], axis=0)
    # if m == 1:
    #     A = A[:,]
    return A, flops


def symFHT(A):
    """
    Faster way of applying FHT from both sides to a symmetric matrix A.
    Assume that n is power of 2.
    """
    n = A.shape[0]
    flops = 0
    if n == 1:
        return A, flops
    A11 = A[:n//2, :n//2]
    A12 = A[:n//2, n//2:]
    A22 = A[n//2:, n//2:]
    
    C1, flops1 = symFHT(A11)
    C3, flops3 = symFHT(A22)
    # indices = np.arange(n//2)
    C2, flops21 = fht(A12)
    C2, flops22 = fht(C2.T)
    C2 = C2.T
    flops += flops1 + flops3 + flops21 + flops22

    B1 = C1 + C2.T
    B2 = C2 + C3
    B3 = C2 + C2.T   # B3 is symmetric
    flops += (n**2 / 2) + (n**2 / 8) + (n / 4)

    M11 = B1 + B2   # M11 is symmetric
    M12 = B1 - B2
    M22 = M11 - 2 * B3   # M22 is symmetric
    flops += (n**2 / 4) + 3 * (n**2 / 8) + 3 * (n / 4)
    return np.block([[M11, M12], [M12.T, M22]]), flops


def sketch_or_subsample(A, k, sketch, lam=0):   # lam controls the weight of uniform sampling
    n = A.shape[0]
    if sketch == 'gaussian':
    # if sketch == True:
        Sf = GaussianSketchFactory((k, n))

    elif sketch == 'row_norms_avg':
        row_norms = np.linalg.norm(A, axis=1)
        avg_norm = np.sum(row_norms) / n
        row_norms += lam * avg_norm
        Sf = SubsamplingSketchFactory(shape=(k, n), probabilities=row_norms)

    elif sketch == 'diag_avg':
        diag = np.diagonal(A)
        diag = diag / np.sum(diag)
        diag += lam / n
        Sf = SubsamplingSketchFactory(shape=(k, n), probabilities=diag)

    elif sketch == 'uniform':
        prob = np.ones(n) / n
        Sf = SubsamplingSketchFactory(shape=(k, n), probabilities=prob)

    return Sf
