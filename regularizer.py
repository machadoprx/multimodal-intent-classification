import numpy as np

def regularization(S, k, X_txt, incomplete_indices, iterations=15, mi=1.0):

    n, dim = X_txt.shape
    X_prop = X_txt.copy()
    eps = 1e-5

    scale_sum_repeated = 1.0 / (np.sum(S, axis=1) + eps)
    scale_sum_repeated = np.reshape(np.repeat(scale_sum_repeated,repeats=[dim]), (n,dim))

    for it in range(iterations):
        X_prop_old = X_prop.copy()
        X_prop = (1.0 - mi) * X_txt + mi * (np.matmul(S, X_prop) * scale_sum_repeated)
        energy = np.mean(np.linalg.norm(X_prop-X_prop_old, axis=1))
        print(f'Iter {it} | Energy {energy}')
        del X_prop_old

    return X_prop