import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.datasets import (make_low_rank_matrix)

from scipy.linalg import svd, sqrtm, cholesky, solve_triangular
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.preprocessing import StandardScaler

from sketch import SubsamplingSketchFactory   # Sketch, SketchFactory, GaussianSketchFactory
from utils import fht
from kaczmarz import kaczmarz_plusplus


def load_dataset(m, n, effective_rank=100):
    """Load and preprocess the dataset."""
    X = make_low_rank_matrix(n_samples=m, n_features=n, effective_rank=effective_rank,tail_strength=0.01)
    return X

def run_coordinate_descent(A, b, x, x0, t_max, sol_norm, Sf_list, maxiter=8, metric="residual"):
    """Run CD++ with/without acceleration, with/without block memoization."""
    num_runs = len(Sf_list)   # Number of multiple runs
    dists2_kz_runs = []
    dists2_kz_acc_runs = []
    dists2_kz_block_runs = []
    dists2_kzpp_runs = []

    for run_idx in range(num_runs):
        Sf = Sf_list[run_idx]

        X_kz = kaczmarz_plusplus(A, b, x0, Sf, t_max, accelerated=False, block=False, exact=False, reg=1e-8, acc=1e-1, maxiter=maxiter)
        X_kz_acc = kaczmarz_plusplus(A, b, x0, Sf, t_max, accelerated=True, block=False, exact=False, reg=1e-8, acc=1e-1, maxiter=maxiter)
        X_kz_block = kaczmarz_plusplus(A, b, x0, Sf, t_max, accelerated=False, block=True, exact=False, reg=1e-8, acc=1e-1, maxiter=maxiter)
        X_kzpp = kaczmarz_plusplus(A, b, x0, Sf, t_max, accelerated=True, block=True, exact=False, reg=1e-8, acc=1e-1, maxiter=maxiter)

        if metric == "l2-norm":
            dists2_kz = (1/sol_norm) * np.linalg.norm(X_kz - x[None, :], axis=1) ** 2
            dists2_kz_acc = (1/sol_norm) * np.linalg.norm(X_kz_acc - x[None, :], axis=1) ** 2
            dists2_kz_block = (1/sol_norm) * np.linalg.norm(X_kz_block - x[None, :], axis=1) ** 2
            dists2_kzpp = (1/sol_norm) * np.linalg.norm(X_kzpp - x[None, :], axis=1) ** 2

        elif metric == "residual":
            dists2_kz = (1/sol_norm) * np.linalg.norm(X_kz @ A.T - b[None, :], axis=1)
            dists2_kz_acc = (1/sol_norm) * np.linalg.norm(X_kz_acc @ A.T - b[None, :], axis=1)
            dists2_kz_block = (1/sol_norm) * np.linalg.norm(X_kz_block @ A.T - b[None, :], axis=1)
            dists2_kzpp = (1/sol_norm) * np.linalg.norm(X_kzpp @ A.T - b[None, :], axis=1)

        dists2_kz_runs.append(dists2_kz)
        dists2_kz_acc_runs.append(dists2_kz_acc)
        dists2_kz_block_runs.append(dists2_kz_block)
        dists2_kzpp_runs.append(dists2_kzpp)

    ### Average over multiple runs
    dists2_kz_avg = np.mean(dists2_kz_runs, axis=0)
    dists2_kz_acc_avg = np.mean(dists2_kz_acc_runs, axis=0)
    dists2_kz_block_avg = np.mean(dists2_kz_block_runs, axis=0)
    dists2_kzpp_avg = np.mean(dists2_kzpp_runs, axis=0)

    return dists2_kz_avg, dists2_kz_acc_avg, dists2_kz_block_avg, dists2_kzpp_avg


def plot_results(
    t_max_list,
    dists2_kz_list,
    dists2_kz_acc_list,
    dists2_kz_block_list,
    dists2_kzpp_list,
    k_list,
    filename,
    effective_rank
):
    """Plot and save the results."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(len(k_list) * 5, 5))

    ### Plot convergence vs iterations

    for i in range(len(k_list)):
        plt.subplot(1, len(k_list), i+1)
        t_max = t_max_list[i]
        dists2_kz = dists2_kz_list[i]
        dists2_kz_acc = dists2_kz_acc_list[i]
        dists2_kz_block = dists2_kz_block_list[i]
        k, dists2_kzpp = dists2_kzpp_list[i]

        ts = np.arange(t_max + 1)
        plt.semilogy(ts, dists2_kz, label="Kaczmarz", color="darkorange")
        plt.semilogy(ts, dists2_kz_block, label="K++ w/o Accel", color="crimson")
        plt.semilogy(ts, dists2_kz_acc, label="K++ w/o Memo", color="turquoise")   # , linestyle="--"
        plt.semilogy(ts, dists2_kzpp, label="Full K++", color="royalblue")
        plt.xlabel("Iterations", fontsize=15)
        plt.ylabel("Residual $\|A x_t - b\| / \|b\|$", fontsize=15)
        # plt.ylabel("Squared distance to solution $\|x_t - x\|_A^2 / \|x\|_A^2$")
        plt.title(f"block size={k_list[i]}", fontsize=15)
        # plt.ylim(1e-7, 1e0)
        plt.legend(fontsize="16", loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Overall title
    plt.suptitle(f"Effective rank: {effective_rank}", fontsize=20)

    plt.savefig(filename)
    plt.close()


def main():
    dataset_name = "low_rank"
    m = 4096
    n = 1024
    num_runs = 10   # do multiple runs and take average
    mode = "write"   # or "read"
    metric = "residual"
    np.random.seed(0)

    effective_rank_list = [50, 100]
    k_list = [100, 200]   # [50, 100, 150, 200]

    for effective_rank in effective_rank_list:
        t_max_list = []
        dists2_kz_list = []
        dists2_kz_acc_list = []
        dists2_kz_block_list = []
        dists2_kzpp_list = []
        maxiter = 8
        for k in k_list:
            print(f"Dataset: {dataset_name}, effective rank: {effective_rank}, k: {k}")
            A = load_dataset(m, n, effective_rank)

            t_max = int(400000 / k)
            t_max_list.append(t_max)

            ### RHT: diagonal step
            random_signs = np.random.choice([-1, 1], size=m)
            D = np.diag(random_signs)
            A = D @ A

            ### RHT: Hadamard transform step
            A, flops_rht = fht(A)
            x_ = np.random.randn(n)
            b = A @ x_
            x0 = np.zeros(n)
            sol_norm = np.linalg.norm(b)

            if metric == "l2-norm":
                sol_norm = np.linalg.norm(x_) ** 2

            Sf_list = [SubsamplingSketchFactory((k, m)) for _ in range(num_runs)]

            if mode == "write":
                dists2_kz, dists2_kz_acc, dists2_kz_block, dists2_kzpp = run_coordinate_descent(A, b, x_, x0, t_max, sol_norm, Sf_list, maxiter, metric=metric)
                dists2_kz_list.append(dists2_kz)
                dists2_kz_acc_list.append(dists2_kz_acc)
                dists2_kz_block_list.append(dists2_kz_block)
                dists2_kzpp_list.append((k, dists2_kzpp))

        ### Save the distance data
        if mode == "write":
            with open(f'kz_acc_{dataset_name}_{effective_rank}.npy', 'wb') as f:
                for i in range(len(k_list)):
                    np.save(f, dists2_kz_list[i])
                    np.save(f, dists2_kz_acc_list[i])
                    np.save(f, dists2_kz_block_list[i])
                    np.save(f, dists2_kzpp_list[i][1])
                                
        ### Open the distance data, uncomment for revision of plots
        elif mode == "read":
            with open(f'kz_acc_{dataset_name}_{effective_rank}.npy', 'rb') as f:
                for i in range(len(k_list)):
                    dists2_kz_list.append(np.load(f))
                    dists2_kz_acc_list.append(np.load(f))
                    dists2_kz_block_list.append(np.load(f))
                    dists2_kzpp_list.append((k, np.load(f)))



        ### Plot and save results
        filename = f"kz_acc_{dataset_name}_{effective_rank}.png"
        plot_results(
            t_max_list,
            dists2_kz_list,
            dists2_kz_acc_list,
            dists2_kz_block_list,
            dists2_kzpp_list,
            k_list,
            filename,
            effective_rank
        )

if __name__ == "__main__":
    main()
