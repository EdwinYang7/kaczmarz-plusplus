import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.linalg import svd

from sketch import SubsamplingSketchFactory
from utils import symFHT
from kaczmarz import coordinate_descent_meta
from psd_flops import load_dataset, compute_kernel, setup_system


def run_coordinate_descent(A, b, x, x0, t_max, sA, sol_norm, k, Sf_list, metric="residual"):
    num_runs = len(Sf_list)
    # dists2_cd_runs = []
    # dists2_cd_acc_runs = []
    # dists2_cd_block_runs = []
    dists2_cd_block_acc_runs = []

    for run_idx in range(num_runs):
        Sf = Sf_list[run_idx]

        # X_cd, flops_cd = coordinate_descent_meta(A, b, x0, Sf, t_max, accelerated=False, block=False)
        # X_cd_acc, flops_cd_acc = coordinate_descent_meta(A, b, x0, Sf, t_max, block=False)
        # X_block, flops_block = coordinate_descent_meta(A, b, x0, Sf, t_max, accelerated=False)
        X_block_acc, flops_block_acc = coordinate_descent_meta(A, b, x0, Sf, t_max)

        if metric == "A-norm":
            # dists2_cd = (1/sol_norm) * np.linalg.norm((X_cd - x[None, :]) @ sA, axis=1) ** 2
            # dists2_cd_acc = (1/sol_norm) * np.linalg.norm((X_cd_acc - x[None, :]) @ sA, axis=1) ** 2
            # dists2_cd_block = (1/sol_norm) * np.linalg.norm((X_block - x[None, :]) @ sA, axis=1) ** 2
            dists2_cd_block_acc = (1/sol_norm) * np.linalg.norm((X_block_acc - x[None, :]) @ sA, axis=1) ** 2

        elif metric == "residual":
            # dists2_cd = (1/sol_norm) * np.linalg.norm(X_cd @ A - b[None, :], axis=1)
            # dists2_cd_acc = (1/sol_norm) * np.linalg.norm(X_cd_acc @ A - b[None, :], axis=1)
            # dists2_cd_block = (1/sol_norm) * np.linalg.norm(X_block @ A - b[None, :], axis=1)
            dists2_cd_block_acc = (1/sol_norm) * np.linalg.norm(X_block_acc @ A - b[None, :], axis=1)

        # dists2_cd_runs.append(dists2_cd)
        # dists2_cd_acc_runs.append(dists2_cd_acc)
        # dists2_cd_block_runs.append(dists2_cd_block)
        dists2_cd_block_acc_runs.append(dists2_cd_block_acc)

    ### Average over runs

    # dists2_cd_avg = np.mean(dists2_cd_runs, axis=0)
    # dists2_cd_acc_avg = np.mean(dists2_cd_acc_runs, axis=0)
    # dists2_cd_block_avg = np.mean(dists2_cd_block_runs, axis=0)
    dists2_cd_block_acc_avg = np.mean(dists2_cd_block_acc_runs, axis=0)

    return dists2_cd_block_acc_avg


def run_coordinate_descent_with_reg(A, b, x, x0, t_max, sA, sol_norm, Sf_list, reg_values, metric="residual"):
    num_runs = len(Sf_list)
    dists2_cd_reg = []
    for reg in reg_values:
        dists2_cd_runs = []
        for run_idx in range(num_runs):
            Sf = Sf_list[run_idx]
            X_cd_reg, flops = coordinate_descent_meta(A, b, x0, Sf, t_max, reg=reg)
            if metric == "A-norm":
                dist2 = (1 / sol_norm) * np.linalg.norm((X_cd_reg - x[None, :]) @ sA, axis=1) ** 2
            elif metric == "residual":
                dist2 = (1 / sol_norm) * np.linalg.norm(X_cd_reg @ A - b[None, :], axis=1)
            dists2_cd_runs.append(dist2)
        
        dist2 = np.mean(dists2_cd_runs, axis=0)
        dists2_cd_reg.append((reg, dist2))
    return dists2_cd_reg


def plot_results(
    t_max,
    dists2_cd_block_acc_list,
    dists2_cd_reg_list,
    name_list,
    n,
    k,
    filename,
    dataset_name,
    kernel_type,
    gamma,
    mu,
    num_runs,
    effective_rank_list = [25, 50, 100, 200]
):
    """Plot and save the results."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(20, 5))

    ts = np.arange(t_max + 1)
    for i in range(4):
        plt.subplot(1, 4, i+1)
        dists2_cd_block_acc = dists2_cd_block_acc_list[i]
        dists2_cd_reg = dists2_cd_reg_list[i]
        kernel_type, gamma = name_list[i]

        for j, (reg, dist2) in enumerate(dists2_cd_reg):
            plt.semilogy(ts, dist2, label=f"reg={reg}", color=color_cycle[j % len(color_cycle)])
    
        plt.semilogy(ts, dists2_cd_block_acc, label="no reg", color=color_cycle[0 % len(color_cycle)], linestyle=":")

        plt.xlabel("Iterations", fontsize=15)
        plt.ylabel("Residual $\|A x_t - b\| / \|b\|$", fontsize=15)
        if dataset_name  == "low_rank":
            effective_rank = effective_rank_list[i]
            plt.title(f"Effective Rank: {effective_rank}", fontsize=15)
        else:
            plt.title(f"kernel: {kernel_type}, width: {gamma}", fontsize=15)
        plt.ylim(1e-7, 1e-2)
        plt.legend(fontsize=14, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    ### Overall title including dataset, kernel, mu

    plt.suptitle(
        f"Dataset: {dataset_name}",
        fontsize=20,
    )

    plt.savefig(filename)
    plt.close()


def main():
    datasets = ["abalone", "phoneme", "california_housing", "covtype", "low_rank"]
    kernel_types = ["gaussian", "laplacian"]
    d = 4096
    m = d
    n = d
    mu = 1e-3
    k = 200
    num_runs = 1  # 5
    metric = "residual"   # or "A-norm"
    np.random.seed(0)

    reg_values = np.logspace(-2, -10, num=5)

    for dataset_name in datasets:
        dists2_cd_reg_list = []
        dists2_cd_block_acc_list = []
        name_list = []
        effective_rank_list = [25, 50, 100, 200]
        for kernel_type in kernel_types:
            for gamma in [1e-1, 1e-2]:
                name_list.append((kernel_type, gamma))
                if dataset_name == "low_rank":
                    effective_rank = effective_rank_list.pop(0)
                    print(f"Running on dataset: {dataset_name}, effective rank: {effective_rank}, regularization: 1e-3")
                    A = load_dataset(dataset_name, d, effective_rank)
                else:
                    print(f"Running on dataset: {dataset_name}, kernel: {kernel_type}, width: {gamma}, regularization: {mu}")
                    X_normalized, b = load_dataset(dataset_name, d)
                    A0 = compute_kernel(X_normalized, kernel_type, gamma=gamma)
                    A = setup_system(A0, mu)

                t_max = int(300000 / k)
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
                    ol_norm = np.linalg.norm(sA @ x_) ** 2

                Sf_list = [SubsamplingSketchFactory((k, n)) for _ in range(num_runs)]
            
                dists2_cd_block_acc = run_coordinate_descent(A, b, x_, x0, t_max, sA, sol_norm, k, Sf_list, metric=metric)
                dists2_cd_reg = run_coordinate_descent_with_reg(A, b, x_, x0, t_max, sA, sol_norm, Sf_list, reg_values, metric=metric)
                dists2_cd_reg_list.append(dists2_cd_reg)
                dists2_cd_block_acc_list.append(dists2_cd_block_acc)

        filename = f"reg_{dataset_name}.png"
        plot_results(
            t_max,
            dists2_cd_block_acc_list,
            dists2_cd_reg_list,
            name_list,
            n,
            k,
            filename,
            dataset_name,
            kernel_type,
            gamma,
            mu,
            num_runs,
        )

if __name__ == "__main__":
    main()
