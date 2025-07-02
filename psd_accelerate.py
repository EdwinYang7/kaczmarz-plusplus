import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    fetch_openml,
    make_low_rank_matrix
)

from scipy.linalg import svd, sqrtm, cholesky, solve_triangular
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.preprocessing import StandardScaler

from sketch import SubsamplingSketchFactory   # Sketch, SketchFactory, GaussianSketchFactory
from utils import symFHT
from kaczmarz import coordinate_descent_meta


def load_dataset(name, d, effective_rank=100):
    """Load and preprocess the dataset."""
    if name == "california_housing":
        data = fetch_california_housing()
    elif name == "covtype":
        data = fetch_covtype()
    elif name == "abalone":
        data = fetch_openml(data_id=720, as_frame=False, parser="liac-arff")
    elif name == "phoneme":
        data = fetch_openml(data_id=1489, as_frame=False)

    if name == "low_rank":   # Synthetic low rank dataset
        X = make_low_rank_matrix(n_samples=d, n_features=d, effective_rank=effective_rank,tail_strength=0.01)
        A = X @ X.T + 1e-3 * np.eye(d)
        return A
    
    else:   # Benchmark dataset
        X, y = data.data, data.target
        X = X[:d, :]
        b = y[:d]
        # b = b / np.linalg.norm(b)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized, b


def compute_kernel(X, kernel_type, **kwargs):
    """Construct kernel matrix for benchmark dataset."""
    if kernel_type == "gaussian":
        gamma = kwargs.get("gamma", 1e-1)
        A0 = rbf_kernel(X, gamma=gamma)
    elif kernel_type == "laplacian":
        gamma = kwargs.get("gamma", 1e-1)
        A0 = laplacian_kernel(X, gamma=gamma)
    return A0


def setup_system(A0, mu):
    """Set up regularized linear system for benchmark dataset."""
    n = A0.shape[0]
    A1 = mu * np.eye(n)
    A = A0 + A1
    return A


def run_coordinate_descent(A, b, x, x0, t_max, sA, sol_norm, Sf_list, metric="residual"):
    """Run CD++ with/without acceleration, with/without block memoization."""
    num_runs = len(Sf_list)   # Number of multiple runs
    dists2_cd_runs = []
    dists2_cd_acc_runs = []
    dists2_cd_block_runs = []
    dists2_cd_block_acc_runs = []

    for run_idx in range(num_runs):
        Sf = Sf_list[run_idx]

        X_cd, flops_cd = coordinate_descent_meta(A, b, x0, Sf, t_max, accelerated=False, block=False, reg=1e-8)
        X_cd_acc, flops_cd_acc = coordinate_descent_meta(A, b, x0, Sf, t_max, block=False, reg=1e-8)
        X_block, flops_block = coordinate_descent_meta(A, b, x0, Sf, t_max, accelerated=False, reg=1e-8)
        X_block_acc, flops_block_acc = coordinate_descent_meta(A, b, x0, Sf, t_max, reg=1e-8)

        if metric == "A-norm": 
            dists2_cd = (1/sol_norm) * np.linalg.norm((X_cd - x[None, :]) @ sA, axis=1) ** 2
            dists2_cd_acc = (1/sol_norm) * np.linalg.norm((X_cd_acc - x[None, :]) @ sA, axis=1) ** 2
            dists2_cd_block = (1/sol_norm) * np.linalg.norm((X_block - x[None, :]) @ sA, axis=1) ** 2
            dists2_cd_block_acc = (1/sol_norm) * np.linalg.norm((X_block_acc - x[None, :]) @ sA, axis=1) ** 2

        elif metric == "residual":
            dists2_cd = (1/sol_norm) * np.linalg.norm(X_cd @ A - b[None, :], axis=1)
            dists2_cd_acc = (1/sol_norm) * np.linalg.norm(X_cd_acc @ A - b[None, :], axis=1)
            dists2_cd_block = (1/sol_norm) * np.linalg.norm(X_block @ A - b[None, :], axis=1)
            dists2_cd_block_acc = (1/sol_norm) * np.linalg.norm(X_block_acc @ A - b[None, :], axis=1)

        dists2_cd_runs.append(dists2_cd)
        dists2_cd_acc_runs.append(dists2_cd_acc)
        dists2_cd_block_runs.append(dists2_cd_block)
        dists2_cd_block_acc_runs.append(dists2_cd_block_acc)

    ### Average over multiple runs
    dists2_cd_avg = np.mean(dists2_cd_runs, axis=0)
    dists2_cd_acc_avg = np.mean(dists2_cd_acc_runs, axis=0)
    dists2_cd_block_avg = np.mean(dists2_cd_block_runs, axis=0)
    dists2_cd_block_acc_avg = np.mean(dists2_cd_block_acc_runs, axis=0)

    return dists2_cd_avg, dists2_cd_acc_avg, dists2_cd_block_avg, dists2_cd_block_acc_avg


def plot_cdpp_block_sizes(
    t_max_list,
    dists2_cd_block_acc_list,
    k_list,
    filename,
    dataset_name,
    kernel_type=None,
    gamma=None,
    effective_rank=None,
):
    """Testing CD++ for different block sizes."""
    plt.figure(figsize=(10, 6))

    for i, k in enumerate(k_list):
        t_max = t_max_list[i]
        ts = np.arange(t_max + 1)
        _, dists2 = dists2_cd_block_acc_list[i]
        plt.semilogy(ts, dists2, label=f"block size = {k}", linewidth=2.5)

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Residual $\|A x_t - b\| / \|b\|$", fontsize=14)
    plt.ylim(1e-7, 1e0)
    plt.legend(title="CD++ settings", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    if dataset_name == "low_rank":
        plt.title(f"CD++ on low-rank (eff. rank={effective_rank})", fontsize=16)
    else:
        plt.title(f"CD++ on {dataset_name}, kernel={kernel_type}, γ={gamma}", fontsize=16)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_results(
    t_max_list,
    dists2_cd_list,
    dists2_cd_acc_list,
    dists2_cd_block_list,
    dists2_cd_block_acc_list,
    # kappas,
    k_list,
    filename,
    dataset_name,
    kernel_type,
    gamma,
    # sketch_method,
    effective_rank
):
    """Plot and save the results."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(20, 5))

    ### Plot convergence vs iterations

    for i in range(4):
        plt.subplot(1, 4, i+1)
        t_max = t_max_list[i]
        dists2_cd = dists2_cd_list[i]
        dists2_cd_acc = dists2_cd_acc_list[i]
        dists2_cd_block = dists2_cd_block_list[i]
        k, dists2_cd_block_acc = dists2_cd_block_acc_list[i]

        ts = np.arange(t_max + 1)
        plt.semilogy(ts, dists2_cd, label="CD", color="darkorange")
        plt.semilogy(ts, dists2_cd_block, label="CD+Memo", color="crimson")
        plt.semilogy(ts, dists2_cd_acc, label="CD+Accel", color="turquoise")   # , linestyle="--"
        plt.semilogy(ts, dists2_cd_block_acc, label="Full CD++", color="royalblue")   # , linestyle=":"
        plt.xlabel("Iterations", fontsize=15)
        plt.ylabel("Residual $\|A x_t - b\| / \|b\|$", fontsize=15)
        # plt.ylabel("Squared distance to solution $\|x_t - x\|_A^2 / \|x\|_A^2$")
        plt.title(f"block size={k_list[i]}", fontsize=15)
        plt.ylim(1e-7, 1e0)
        plt.legend(fontsize="16", loc="upper right")

        # x_marker = ts[int(t_max / 2)]
        # y_marker = dists2_cd_block_acc[int(t_max / 2)]
        # plt.plot(x_marker, y_marker, 'x', color='royalblue', markersize=8, label='Marker')

    ### Plot singular value decay

    # plt.subplot(1, 4, 4)
    # plt.semilogy(range(2 * k),kappas[0:2*k])
    # plt.xlabel("Index")
    # plt.ylabel("Tail condition number $\kappa_k$")
    # plt.title("Singular value decay")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Overall title
    if dataset_name  == "low_rank":
        plt.suptitle(
            f"Effective rank: {effective_rank}",
            fontsize=20,   # Sketch: {sketch_method}
        )
    else:
        plt.suptitle(
            f"Dataset: {dataset_name}, kernel: {kernel_type} with width={gamma}",
            fontsize=20,   # Sketch: {sketch_method}
        )

    plt.savefig(filename)
    plt.close()


def main():
    datasets = ["abalone"]   # "low_rank"
    kernel_types = ["gaussian", "laplacian"]
    d = 4096
    m = d
    n = d
    num_runs = 10   # do multiple runs and take average
    effective_rank = 200   # 25, 50, 100, 200
    mu = 1e-3
    mode = "read"   # or "write"
    metric = "residual"   # or "A-norm"
    np.random.seed(0)

    for dataset_name in datasets:
        for kernel_type in kernel_types:
            for gamma in [1e-1, 1e-2]:
                k_list = [25, 50, 100, 200]
                t_max_list = []
                dists2_cd_list = []
                dists2_cd_acc_list = []
                dists2_cd_block_list = []
                dists2_cd_block_acc_list = []
                for k in k_list:
                    if dataset_name == "low_rank":
                        print(f"Running on dataset: {dataset_name}, effective rank: {effective_rank}")
                        A = load_dataset(dataset_name, d, effective_rank)
                    else:
                        print(f"Running on dataset: {dataset_name}, kernel: {kernel_type}, kernel width: {gamma}, sketch size: {k}")
                        X_normalized, b = load_dataset(dataset_name, d)
                        A0 = compute_kernel(X_normalized, kernel_type, gamma=gamma)
                        A = setup_system(A0, mu)

                    t_max = int(400000 / k)
                    t_max_list.append(t_max)
                    n = A.shape[0]

                    ### RHT: diagonal step
                    random_signs = np.random.choice([-1, 1], size=n)
                    D = np.diag(random_signs)
                    A = D @ A @ D

                    ### RHT: Hadamard transform step
                    A, flops_rht = symFHT(A)
                    x_ = np.random.randn(n)
                    b = A @ x_
                    x0 = np.zeros(n)

                    sA = 0
                    sol_norm = np.linalg.norm(b)

                    if metric == "A-norm":
                        U, s, VT = svd(A)
                        kappas = (1 / s[-1]) * s
                        sA = U @ np.diag(np.sqrt(s)) @ VT
                        sol_norm = np.linalg.norm(sA @ x_) ** 2

                    Sf_list = [SubsamplingSketchFactory((k, n)) for _ in range(num_runs)]

                    if mode == "write":
                        dists2_cd, dists2_cd_acc, dists2_cd_block, dists2_cd_block_acc = run_coordinate_descent(A, b, x_, x0, t_max, sA, sol_norm, Sf_list, metric="residual")
                        dists2_cd_list.append(dists2_cd)
                        dists2_cd_acc_list.append(dists2_cd_acc)
                        dists2_cd_block_list.append(dists2_cd_block)
                        dists2_cd_block_acc_list.append((k, dists2_cd_block_acc))

                ### Save the distance data
                if mode == "write":
                    if dataset_name == "low_rank":
                        with open(f'acc_{dataset_name}_{effective_rank}.npy', 'wb') as f:
                            for i in range(4):
                                np.save(f, dists2_cd_list[i])
                                np.save(f, dists2_cd_acc_list[i])
                                np.save(f, dists2_cd_block_list[i])
                                np.save(f, dists2_cd_block_acc_list[i][1])
                
                    else:
                        with open(f'acc_{dataset_name}_{kernel_type}_{gamma}.npy', 'wb') as f:
                            for i in range(4):
                                np.save(f, dists2_cd_list[i])
                                np.save(f, dists2_cd_acc_list[i])
                                np.save(f, dists2_cd_block_list[i])
                                np.save(f, dists2_cd_block_acc_list[i][1])
                                
                ### Open the distance data, uncomment for revision of plots
                elif mode == "read":
                    if dataset_name == "low_rank":
                        with open(f'acc_{dataset_name}_{effective_rank}.npy', 'rb') as f:
                            for i in range(4):
                                dists2_cd_list.append(np.load(f))
                                dists2_cd_acc_list.append(np.load(f))
                                dists2_cd_block_list.append(np.load(f))
                                dists2_cd_block_acc_list.append((k, np.load(f)))

                    else:
                        with open(f'acc_{dataset_name}_{kernel_type}_{gamma}.npy', 'rb') as f:
                            for i in range(4):
                                dists2_cd_list.append(np.load(f))
                                dists2_cd_acc_list.append(np.load(f))
                                dists2_cd_block_list.append(np.load(f))
                                dists2_cd_block_acc_list.append((k, np.load(f)))


                ### Plot and save results
                if dataset_name == "low_rank":
                    filename = f"acc_{dataset_name}_{effective_rank}.png"
                else:
                    filename = f"acc_{dataset_name}_{kernel_type}_{gamma}.png"
                # plot_results(
                #     t_max_list,
                #     dists2_cd_list,
                #     dists2_cd_acc_list,
                #     dists2_cd_block_list,
                #     dists2_cd_block_acc_list,
                #     # kappas,
                #     k_list,
                #     filename,
                #     dataset_name,
                #     kernel_type,
                #     gamma,
                #     # sketch_method,
                #     effective_rank
                # )

                plot_cdpp_block_sizes(
    t_max_list,
    dists2_cd_block_acc_list,
    k_list,
    filename,
    dataset_name,
    kernel_type=kernel_type,
    gamma=gamma,
    effective_rank=effective_rank,
)

if __name__ == "__main__":
    main()
