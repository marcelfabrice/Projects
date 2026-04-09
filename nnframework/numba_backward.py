from numba import njit
import numpy as np

# Numba-optimierte Kernfunktion für Conv2d.backward
@njit
def conv2d_backward_core(delta, X, kernels):
    """
    NumPy+Numba-Version des rechenintensiven Teils von Conv2d.backward.

    delta   : (H_out, W_out, n_out)
    X       : (H, W, C_in)
    kernels : (n_out, C_in, K, K)

    return:
        delta_prev : (H, W, C_in)
        grad       : (n_out, C_in, K, K)
    """
    H, W, C_in = X.shape
    n_out, n_in, K, _ = kernels.shape
    H_out, W_out, _ = delta.shape

    grad = np.zeros_like(kernels)
    delta_prev = np.zeros_like(X)

    # Über alle Filter (Output-Channels) iterieren
    for f in range(n_out):
        dB = delta[:, :, f]      # (H_out, W_out)
        Kf = kernels[f]          # (C_in, K, K)

        # 1) Gradient für Kernel f berechnen
        for c in range(C_in):
            for u in range(K):
                for v in range(K):
                    s = 0.0
                    for i in range(H_out):
                        for j in range(W_out):
                            s += dB[i, j] * X[i + u, j + v, c]
                    grad[f, c, u, v] = s

        # 2) delta_prev berechnen: Faltung von dB mit rotiertem Kernel (180° gedreht)
        for c in range(C_in):
            for i in range(H):
                for j in range(W):
                    s = 0.0
                    for u in range(K):
                        ii = i - u
                        if ii < 0 or ii >= H_out:
                            continue
                        for v in range(K):
                            jj = j - v
                            if jj < 0 or jj >= W_out:
                                continue
                            # Manuelle 180°-Rotation des Kernels:
                            s += dB[ii, jj] * Kf[c, K - 1 - u, K - 1 - v]
                    delta_prev[i, j, c] += s

    return delta_prev, grad