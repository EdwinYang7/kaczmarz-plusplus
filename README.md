## Kaczmarz++
This is the public code repository for *Randomized Kaczmarz Methods with Beyond-Krylov Convergence*, a paper by Jiaming Yang, Michał Dereziński, Elizaveta Rebrova and Deanna Needell.
This repository contains Python code to recreate the paper's experiments, as well as the exact experimental data that generated our plots.
+ `psd_accelerate.py` corresponds to the convergence testing experiments in Section 6.2 and Supplement 4.1. It generates Figure 1 in main paper, as well as Figure SM1, SM2, SM3 in supplement.
+ `psd_flops.py` corresponds to the comparison with Krylov subspace methods experiments (in terms of FLOPs) in Section 6.3 and Supplement 4.2. It generates Figure 2 in main paper, as well as Figure SM4 and Table SM1 in supplement.
+ `regularization.py` corresponds to the regularization testing in projection experiments in Supplement 4.3. It generates Figure SM5 in supplement.

For the regeneration of plots, the code assume that all the `.npy` files are stored locally in the same directory as `psd_accelerate.py`, `psd_flops.py` and `regularization.py`.
