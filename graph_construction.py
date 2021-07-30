'''
Furthermore, we use theaverage distance between the top-7nearest neighbors to definethe scale parameter𝜎

Rule 1.Documents with textual features are linked through theirnearest k-neighbors from𝑋𝑇∈R𝑛×𝑑𝑇.
We argue that the BERT semantic representations are moreeffective than visual features for intent classification. 
Thus,our graph-based representation prioritizes the relationshipsof textual similarity between documents when this modalityis present.
Rule 2.Documents with visual features are inserted in the graphusing the nearest k-neighbors from{𝑋𝐼∪𝑋𝐼𝑖𝑛𝑐} ∈R𝑛×𝑑𝐼.
Visual features are used to generate edges between all docu-ments in the data set.
Note that even documents with com-plete modalities can be reached by this rule, thereby gener-ating paths in the graph that allow 
the propagation of textembeddings between the vertices

'''
from sklearn.neighbors import NearestNeighbors
import numpy as np

def gauss_kernel(x, sigma):
    return np.exp(-(x**2)/(2*sigma + 1e-5))

def graph_construction(X_txt, X_img, incomplete_indices, k=30):
    n = X_txt.shape[0]
    A = np.zeros((n, n))

    neigh_im = NearestNeighbors(n_neighbors=k).fit(X_img)
    neigh_txt = NearestNeighbors(n_neighbors=k).fit(X_txt)

    dists_im, neighbors_im = neigh_im.kneighbors(X_img, return_distance=True)
    dists_txt, neighbors_txt = neigh_txt.kneighbors(X_txt, return_distance=True)

    sigma_im = dists_im[:, :7]
    sigma_im = np.mean(sigma_im, axis=1)

    sigma_txt = dists_txt[:, :7]
    sigma_txt = np.mean(sigma_txt, axis=1)

    for node in range(n):
        if node not in incomplete_indices: 
            for neigh_i, dist_i in zip(neighbors_txt[node], dists_txt[node]):
                A[node, neigh_i] = gauss_kernel(dist_i, sigma_txt[node])
                A[neigh_i, node] = gauss_kernel(dist_i, sigma_txt[node])
        else:
            for neigh_i, dist_i in zip(neighbors_im[node], dists_im[node]):
                A[node, neigh_i] = gauss_kernel(dist_i, sigma_im[node])
                A[neigh_i, node] = gauss_kernel(dist_i, sigma_im[node])

    D = np.sum(A, axis=0)
    D = 1/np.sqrt(D + 1e-5)
    D = np.diag(D)

    S = D @ A @ D
    return S