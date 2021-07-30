import numpy as np
from tqdm import tqdm

def regularization(S, k, X_txt, incomplete_indices, iterations=15, mi=1.0):

    n, dim = X_txt.shape
    X_prop = X_txt.copy()
    eps = 1e-5

    scale_sum_repeated = 1.0 / (np.mean(S, axis=1) + eps)
    scale_sum_repeated = np.reshape(np.repeat(scale_sum_repeated,repeats=[dim]), shape=(n,dim))

    pbar = tqdm(range(0,iterations))

    for iteration in pbar:
        X_prop_old = X_prop.copy()
        X_prop = (1.0 - mi) * X_txt + mi * (np.matmul(S, X_prop) * scale_sum_repeated)
        energy = np.mean(np.linalg.norm(X_prop-X_prop_old, axis=1))
        iteration += 1
        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        pbar.set_description(message)
        del X_prop_old

    return X_prop