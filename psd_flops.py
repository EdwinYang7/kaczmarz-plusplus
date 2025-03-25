import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import cg
from scipy.linalg import svd
import pyamg   # for implementation of GMRES

from sketch import SubsamplingSketchFactory
from utils import symFHT
from kaczmarz import coordinate_descent_meta
from psd_accelerate import load_dataset, compute_kernel, setup_system


def run_cg(A, b, x, x0, pass_max, sA, sol_norm, metric="residual", accuracy=1):
    """Run CG and compute FLOPs."""
    n = A.shape[0]
    niter = 0
    X_cg = np.zeros((pass_max, n))

    flops_cg_per_iter = 2 * n**2 + 11 * n
    flops_cg = []

    def cg_callback(xk):
        nonlocal X_cg, niter, flops_cg
        X_cg[niter, :] = xk
        niter += 1
        flops_cg.append(niter * flops_cg_per_iter)

    cg(A, b, x0=x0, maxiter=pass_max, tol=1e-16, callback=cg_callback)
    if metric == "A-norm":
        dists2_cg = (1 / sol_norm) * np.linalg.norm((X_cg[:niter, :] - x[None, :]) @ sA, axis=1) ** 2
    elif metric == "residual":
        dists2_cg = (1 / sol_norm) * np.linalg.norm(X_cg[:niter, :] @ A - b[None, :], axis=1)

    return dists2_cg, np.array(flops_cg)


def run_gmres(A, b, x, x0, pass_max, sA, sol_norm, metric="residual", accuracy=1):
    """Run GMRES and compute FLOPs."""
    n = A.shape[0]
    niter = 0
    X_gmres = np.zeros((pass_max, n))

    def gmres_list(xk):
        nonlocal X_gmres, niter
        X_gmres[niter, :] = xk
        niter += 1
        pass

    pyamg.krylov.gmres(A, b, x0=x0, tol=1e-16, maxiter=pass_max, callback=gmres_list, restrt=None)
    if metric == "A-norm":
        dists2_gmres = (1 / sol_norm) * np.linalg.norm((X_gmres[:niter, :] - x[None, :]) @ sA, axis=1) ** 2
    elif metric == "residual":
        dists2_gmres = (1 / sol_norm) * np.linalg.norm(X_gmres[:niter, :] @ A - b[None,:], axis=1)

    T = np.arange(1, len(dists2_gmres) + 1)
    flops_gmres = 2 * n**2 * T + 4 * n * T * (T + 1)

    return dists2_gmres, flops_gmres


def run_coordinate_descent(A, b, x, x0, t_max, sA, sol_norm, Sf_list, metric="residual", accuracy=1):
    """Run CD++ with/without block memoization and compute FLOPs."""
    num_runs = len(Sf_list)
    dists2_cd_acc_runs = []
    dists2_block_acc_runs = []

    for run_idx in range(num_runs):
        Sf = Sf_list[run_idx]
        
        X_cd_acc, flops_cd_acc = coordinate_descent_meta(A, b, x0, Sf, t_max, block=False, reg=1e-8)
        X_block_acc, flops_block_acc = coordinate_descent_meta(A, b, x0, Sf, t_max, reg=1e-8)

        if metric == "A-norm":
            dists2_cd_acc = (1/sol_norm) * np.linalg.norm((X_cd_acc - x[None, :]) @ sA, axis=1) ** 2
            dists2_block_acc = (1/sol_norm) * np.linalg.norm((X_block_acc - x[None, :]) @ sA, axis=1) ** 2

        elif metric == "residual":
            dists2_cd_acc = (1/sol_norm) * np.linalg.norm(X_cd_acc @ A - b[None, :], axis=1)
            dists2_block_acc = (1/sol_norm) * np.linalg.norm(X_block_acc @ A - b[None, :], axis=1)
        
        dists2_cd_acc_runs.append(dists2_cd_acc)
        dists2_block_acc_runs.append(dists2_block_acc)

    ### Average over runs
    dists2_cd_acc_avg = np.mean(dists2_cd_acc_runs, axis=0)
    dists2_block_acc_avg = np.mean(dists2_block_acc_runs, axis=0)

    return dists2_cd_acc_avg, dists2_block_acc_avg, flops_cd_acc, flops_block_acc


def find_iters(dists2, flops, accuracy):
    """Find the first iteration where error < accuracy."""
    idx = np.where(dists2 < accuracy)[0]
    if idx.size == 0:
        return idx, 0
    else:
        idx = idx[0]
        return idx, flops[idx]


def plot_results(
    # passes,
    cg_list,
    gmres_list,
    cd_acc_list,
    block_acc_list,
    name_list,
    # kappas,
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

    ### Plot convergence vs FLOPs

    for i in range (4):
        plt.subplot(1, 4, i+1)
        dists2_cg, flops_cg = cg_list[i]
        dists2_gmres, flops_gmres = gmres_list[i]
        dists2_cd_acc, flops_cd_acc = cd_acc_list[i]
        dists2_block_acc, flops_block_acc = block_acc_list[i]
        
        kernel_type, gamma = name_list[i]
        # plt.semilogy(flops_cg, dists2_cg, label="CG", color=color_cycle[0])
        # plt.semilogy(flops_gmres, dists2_gmres, label='GMRES', color=color_cycle[5], linestyle=":")
        # plt.semilogy(flops_cd_acc, dists2_cd_acc, label="CD+Accel", color=color_cycle[1])
        # plt.semilogy(flops_block_acc, dists2_block_acc, label="Full CD++", color=color_cycle[2], linestyle=":")
        plt.semilogy(flops_cg, dists2_cg, label="CG", color="darkorange", linewidth=2.5)
        plt.semilogy(flops_gmres, dists2_gmres, label='GMRES', color="crimson", linewidth=2.5)
        plt.semilogy(flops_cd_acc, dists2_cd_acc, label="CD+Accel", color="turquoise", linewidth=2.5)
        plt.semilogy(flops_block_acc, dists2_block_acc, label="Full CD++", color="royalblue", linewidth=2.5)
        plt.xlabel("FLOPs", fontsize=15)
        plt.ylabel("Residual $\|A x_t - b\| / \|b\|$", fontsize=15)
        if dataset_name  == "low_rank":
            effective_rank = effective_rank_list[i]
            plt.title(f"Effective Rank: {effective_rank}", fontsize=15)
        else:
            plt.title(f"kernel: {kernel_type}, width: {gamma}", fontsize=15)

        plt.xlim(left=0, right=9e9)
        if dataset_name=="covtype" and kernel_type=="gaussian" and gamma==0.1:
            plt.xlim(left=0, right=2e10)
        elif dataset_name!="covtype" and kernel_type=="gaussian" and gamma==1e-2:
            plt.xlim(left=0, right=4e9)
        elif kernel_type=="laplacian" and gamma==1e-1:
            plt.xlim(left=0, right=1.2e10)
        plt.ylim(1e-12, 1e1)
        
        if dataset_name=="low_rank" and i==0:
            plt.xlim(left=0, right=6e9)
            plt.ylim(1e-12, 1e1)
        elif dataset_name=="low_rank" and i==3:
            plt.xlim(left=0, right=1.5e10)
            plt.ylim(1e-12, 1e1)
        elif dataset_name=="low_rank":
            plt.xlim(left=0, right=9e9)
            plt.ylim(1e-12, 1e1)

        plt.legend(fontsize="16", loc="upper right")

    ### Plot convergence vs passes

    # plt.subplot(1, 3, 2)
    # plt.semilogy(dists2_cg, label="CG", color=color_cycle[0])
    # plt.semilogy(dists2_gmres, label='GMRES', color=color_cycle[5])
    # # plt.semilogy(passes, dists2_cd, label="CD", color=color_cycle[0])
    # # plt.semilogy(passes, dists2_cd_heuristic_A, label="CD++ (A norm)", color=color_cycle[1], linestyle="--")
    # plt.semilogy(passes, dists2_cd_acc, label="CD++", color=color_cycle[1])
    # plt.semilogy(passes, dists2_cd_block_acc, label="BCD++", color=color_cycle[2])
    # plt.xlabel("Passes over the input matrix")
    # plt.ylabel("Squared distance to solution")
    # plt.title("Convergence of different methods")
    # plt.xlim(right=100)
    # plt.ylim(1e-10, 1e1)
    # plt.legend(loc="upper right")

    # Plot singular value decay
    # plt.subplot(1, 2, 2)
    # plt.semilogy(range(400),kappas[0:400])
    # plt.xlabel("Index")
    # plt.ylabel("Tail condition number $\kappa_k$")
    # plt.title("Singular value decay")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    ### Overall title

    # plt.suptitle(
    #     f"Dataset: {dataset_name}",
    #     # f"Dataset: {dataset_name}, n: {n}, k: {k}, mu: {mu}, runtimes: {num_runs}",
    #     fontsize=20,
    # )

    plt.savefig(filename)
    plt.close()


def main():
    datasets = ["abalone", "phoneme", "california_housing", "covtype", "low_rank"]
    kernel_types = ["gaussian", "laplacian"]
    d = 4096
    m = d
    n = d
    mu = 1e-3
    num_runs = 5
    k = 200
    pass_max = 200
    t_max = int(pass_max * m / k)
    t_max_cd = int(5 * pass_max * m / k)
    accuracy = 1e-8
    mode = "write"   # or "read"
    metric = "residual"   # or "A-norm"
    np.random.seed(0)

    for dataset_name in datasets:
        cg_list = []
        gmres_list = []
        cd_acc_list = []
        block_acc_list = []
        name_list = []
        effective_rank_list = [25, 50, 100, 200]

        if mode == "read":   # Open the distance data
            with open(f'FLOPs_{dataset_name}.npy', 'rb') as f:
                for i in range(4):
                    cg_list.append(np.load(f))
                    gmres_list.append(np.load(f))
                    cd_acc_list.append(np.load(f))
                    block_acc_list.append(np.load(f))
                    

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

                ### RHT: diagonal step
                n = A.shape[0]
                random_signs = np.random.choice([-1, 1], size=n)
                D = np.diag(random_signs)
                A = D @ A @ D
                flops_pre = n**2 / 2 - n / 2

                ### RHT: Hadamard transform step
                A, flops_rht = symFHT(A)
                flops_pre += flops_rht
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
                    dists2_cg, flops_cg = run_cg(A, b, x_, x0, t_max, sA, sol_norm, accuracy=accuracy)
                    dists2_gmres, flops_gmres = run_gmres(A, b, x_, x0, t_max, sA, sol_norm, accuracy=accuracy)
                    dists2_cd_acc, dists2_block_acc, flops_cd_acc, flops_block_acc = run_coordinate_descent(A, b, x_, x0, t_max_cd, sA, sol_norm, Sf_list, accuracy=accuracy)

                    flops_cd_acc += flops_pre
                    flops_block_acc += flops_pre

                    cg_list.append((dists2_cg, flops_cg))
                    gmres_list.append((dists2_gmres, flops_gmres))
                    cd_acc_list.append((dists2_cd_acc, flops_cd_acc))
                    block_acc_list.append((dists2_block_acc, flops_block_acc))

                elif mode == "read":
                    dists2_cg, flops_cg = cg_list.pop(0)
                    dists2_gmres, flops_gmres = gmres_list.pop(0)
                    dists2_cd_acc, flops_cd_acc = cd_acc_list.pop(0)
                    dists2_block_acc, flops_block_acc = block_acc_list.pop(0)

                idx_cg, flops_cg_idx = find_iters(dists2_cg, flops_cg, accuracy)
                idx_gmres, flops_gmres_idx = find_iters(dists2_gmres, flops_gmres, accuracy)
                idx_cd_acc, flops_cd_acc_idx = find_iters(dists2_cd_acc, flops_cd_acc, accuracy)
                idx_block_acc, flops_block_acc_idx = find_iters(dists2_block_acc, flops_block_acc, accuracy)

                print(f"Dataset: {dataset_name}, kernel: {kernel_type}, width: {gamma}, accuracy: {accuracy}, FLOPs for CG: {flops_cg_idx}, FLOPs for GMRES: {flops_gmres_idx}, FLOPs for CD+Accel: {flops_cd_acc_idx}, FLOPs for CD++: {flops_block_acc_idx}")

        if mode == "write":   #  Save the distance data
            with open(f'FLOPs_{dataset_name}.npy', 'wb') as f:
                for i in range(4):
                    np.save(f, cg_list[i])
                    np.save(f, gmres_list[i])
                    np.save(f, cd_acc_list[i])
                    np.save(f, block_acc_list[i])


        ### Plot and save results
        filename = f"FLOPs_{dataset_name}.png"
        plot_results(
            cg_list,
            gmres_list,
            cd_acc_list,
            block_acc_list,
            name_list,
            # kappas,
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
