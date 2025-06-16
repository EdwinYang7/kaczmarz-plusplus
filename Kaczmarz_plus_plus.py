import numpy as np
import random
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.datasets import make_low_rank_matrix

from scipy.linalg import svd, sqrtm, cholesky, solve_triangular
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.preprocessing import StandardScaler

from sketch import SubsamplingSketchFactory
from utils import fht
from kaczmarz import kaczmarz_plusplus


def load_dataset(m, n, effective_rank=100):
    """Load and preprocess the dataset."""
    X = make_low_rank_matrix(n_samples=m, n_features=n, effective_rank=effective_rank,tail_strength=0.01)
    return X


def run_kaczmarz_maxiter(A, b, x, x0, t_max, sol_norm, Sf_list, maxiter_list, metric="residual"):
    """Run kaczmarz++ with inexact inner solver."""
    num_runs = len(Sf_list)
    dists2_kzpp_iter = []

    for maxiter in maxiter_list:
        dists2_kzpp_runs = []
        for run_idx in range(num_runs):
            Sf = Sf_list[run_idx]
            X_kzpp = kaczmarz_plusplus(A, b, x0, Sf, t_max, accelerated=True, block=True, exact=False, reg=1e-8, acc=1e-1, maxiter=maxiter)

            if metric == "l2-norm":
                dists2_kzpp = (1/sol_norm) * np.linalg.norm(X_kzpp - x[None, :], axis=1) ** 2

            elif metric == "residual":
                dists2_kzpp = (1/sol_norm) * np.linalg.norm(X_kzpp @ A.T - b[None, :], axis=1)
            dists2_kzpp_runs.append(dists2_kzpp)

        dists2_kzpp_avg = np.mean(dists2_kzpp_runs, axis=0)
        dists2_kzpp_iter.append((maxiter, dists2_kzpp_avg))

    return dists2_kzpp_iter


def run_kaczmarz(A, b, x, x0, t_max, sol_norm, Sf_list, metric="residual"):
    """Run kaczmarz++ with exact inner solver."""
    num_runs = len(Sf_list)
    dists2_kz_runs = []

    for run_idx in range(num_runs):
        Sf = Sf_list[run_idx]

        X_kz = kaczmarz_plusplus(A, b, x0, Sf, t_max, accelerated=True, block=True, exact=True, reg=1e-8)

        if metric == "l2-norm":
            dists2_kz = (1/sol_norm) * np.linalg.norm(X_kz - x[None, :], axis=1) ** 2

        elif metric == "residual":
            dists2_kz = (1/sol_norm) * np.linalg.norm(X_kz @ A.T - b[None, :], axis=1)

        dists2_kz_runs.append(dists2_kz)

    dists2_kz_avg = np.mean(dists2_kz_runs, axis=0)

    return dists2_kz_avg


def plot_results(
    t_max_list,
    dists2_kz_list,
    dists2_kzpp_list,
    k_list,
    filename,
    effective_rank
):
    """Plot and save the results."""
    # color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # cmap = plt.cm.get_cmap("autumn_r")
    colors = ["gold", "darkorange", "orangered"]
    plt.figure(figsize=(len(k_list) * 5, 5))
    cmap_custom = mpl.colors.LinearSegmentedColormap.from_list(
    "darkYel_orange_red", colors)

    ### Plot convergence vs iterations

    marker_list = ["o", "s", "^", "D", "v", "x", "+"]
    for i in range(len(k_list)):
        plt.subplot(1, len(k_list), i+1)
        t_max = t_max_list[i]
        dists2_kz = dists2_kz_list[i]
        dists2_kzpp = dists2_kzpp_list[i]

        ts = np.arange(t_max + 1)
        N = len(dists2_kzpp)
        marker_positions = np.linspace(0, t_max, num=5, dtype=int).tolist()

        for j, (maxiter, dist2) in enumerate(dists2_kzpp):
            t = j / (N - 1) if N > 1 else 0.0
            color = cmap_custom(t)
            this_marker = marker_list[j % len(marker_list)] 
            plt.semilogy(ts, dist2, label=f"K++(LSQR-{maxiter})", color=color, linewidth=1.5, linestyle="--", marker=this_marker, markersize=8, markevery=marker_positions)
        plt.semilogy(ts, dists2_kz, label="K++(Cholesky)", color="royalblue", linewidth=1.5) #, zorder=1)
        plt.xlabel("Iterations", fontsize=15)
        plt.ylabel("Residual $\|A x_t - b\| / \|b\|$", fontsize=15)
        # plt.ylabel("Squared distance to solution $\|x_t - x\|^2 / \|x\|^2$", fontsize=15)
        plt.title(f"block size={k_list[i]}", fontsize=15)
        plt.legend(fontsize="16", loc="upper right")
    

    # Plot singular value decay
    # plt.subplot(1, 2, 2)
    # plt.semilogy(range(400),kappas[0:400])
    # plt.xlabel("Index")
    # plt.ylabel("Tail condition number $\kappa_k$")
    # plt.title("Singular value decay")

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
    k_list = [100, 200]
    maxiter_list = [2, 4, 8]

    for effective_rank in effective_rank_list:
        t_max_list = []
        dists2_kz_list = []
        dists2_kzpp_list = []
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

            # U, s, VT = svd(A)
            # kappas = (1 / s[-1]) * s

            Sf_list = [SubsamplingSketchFactory((k, m)) for _ in range(num_runs)]

            if mode == "write":
                dists2_kz = run_kaczmarz(A, b, x_, x0, t_max, sol_norm, Sf_list,metric=metric)
                dists2_kzpp_iter = run_kaczmarz_maxiter(A, b, x_, x0, t_max, sol_norm, Sf_list, maxiter_list, metric=metric)
                dists2_kz_list.append(dists2_kz)
                dists2_kzpp_list.append(dists2_kzpp_iter)
 
        ### Save the distance data
        # if mode == "write":
        #     with open(f'kz_{dataset_name}_{effective_rank}.npy', 'wb') as f:
        #         for i in range(len(k_list)):
        #             np.save(f, dists2_kz_list[i])
        #             for j in range(len(maxiter_list)):
        #                 np.save(f, dists2_kzpp_list[i][j])
                                
        ### Open the distance data, uncomment for revision of plots
        # elif mode == "read":
        #     with open(f'kz_{dataset_name}_{effective_rank}.npy', 'rb') as f:
        #         for i in range(len(k_list)):
        #             dists2_kz_list.append(np.load(f))
        #             for j in range(len(maxiter_list)):
        #                 dists2_kzpp_list.append(np.load(f))


        ### Plot and save results
        filename = f"kzpp_{dataset_name}_{effective_rank}_{num_runs}.png"
        plot_results(
            t_max_list,
            dists2_kz_list,
            dists2_kzpp_list,
            k_list,
            filename,
            effective_rank
        )

if __name__ == "__main__":
    main()
