from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.linalg import eigh, eigvalsh

def getEuclideanDistance(
    specEmbA: torch.Tensor, specEmbB: torch.Tensor, device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Calculate Euclidean distances from the given feature tensors.

    Args:
        specEmbA (Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).
        specEmbB (Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).

    Returns:
        dis (Tensor):
            Euclidean distance values of the two sets of spectral embedding vectors.
    """
    specEmbA, specEmbB = specEmbA.to(device), specEmbB.to(device)
    A, B = specEmbA.unsqueeze(dim=1), specEmbB.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


def kmeans_plusplus_torch(
    X: torch.Tensor,
    n_clusters: int,
    random_state: int,
    n_local_trials: int = 30,
    device: torch.device = torch.device('cpu'),
):
    """
    
    Args:
        X (Tensor):
            Matrix containing cosine similarity values among embedding vectors (N x N)
        n_clusters (int):
            Maximum number of speakers for estimating number of speakers.
            Shows stable performance under 20.
        random_state (int):
            Seed variable for setting up a random state.
        n_local_trials (int):
            Number of trials for creating initial values of the center points.
        device (torch.device)
            Torch device variable.

    Returns:
        centers (Tensor):
            The coordinates for center points that are used for initializing k-means algorithm.
        indices (Tensor):
            The indices of the best candidate center points.
    """
    torch.manual_seed(random_state)
    X = X.to(device)
    n_samples, n_features = X.shape

    centers = torch.zeros(n_clusters, n_features, dtype=X.dtype)
    center_id = torch.randint(0, n_samples, (1,)).long()
    indices = torch.full([n_clusters,], -1, dtype=torch.int)

    centers[0] = X[center_id].squeeze(0)
    indices[0] = center_id.squeeze(0)

    centers = centers.to(device)
    closest_dist_diff = centers[0, None].repeat(1, X.shape[0]).view(X.shape[0], -1) - X
    closest_dist_sq = closest_dist_diff.pow(2).sum(dim=1).unsqueeze(dim=0)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials) * current_pot.item()

        if len(closest_dist_sq.shape) > 1:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=1)[0]
        else:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=0)

        candidate_ids = torch.searchsorted(torch_cumsum, rand_vals.to(device))

        N_ci = candidate_ids.shape[0]
        distance_diff = X[candidate_ids].repeat(1, X.shape[0]).view(X.shape[0] * N_ci, -1) - X.repeat(N_ci, 1)
        distance = distance_diff.pow(2).sum(dim=1).view(N_ci, -1)
        distance_to_candidates = torch.minimum(closest_dist_sq, distance)
        candidates_pot = distance_to_candidates.sum(dim=1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices

def kmeans_torch(
    X: torch.Tensor,
    num_clusters: int,
    threshold: float = 1e-4,
    iter_limit: int = 15,
    random_state: int = 0,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Args:
        X (Tensor):
            Cosine similarity matrix calculated from speaker embeddings
        num_clusters (int):
            The estimated number of speakers.
        threshold (float):
            This threshold limits the change of center values. If the square of
            the center shift values are bigger than this threshold, the iteration stops.
        iter_limit (int):
            The maximum number of iterations that is allowed by the k-means algorithm.
        device (torch.device):
            Torch device variable

    Returns:
        selected_cluster_indices (Tensor):
            The assigned cluster labels from the k-means clustering.
    """
    # Convert tensor type to float
    X = X.float().to(device)
    input_size = X.shape[0]

    # Initialize the cluster centers with kmeans_plusplus algorithm.
    plusplus_init_states = kmeans_plusplus_torch(X, n_clusters=num_clusters, random_state=random_state, device=device)
    centers = plusplus_init_states[0]

    selected_cluster_indices = torch.zeros(input_size).long()

    for iter_count in range(iter_limit):
        euc_dist = getEuclideanDistance(X, centers, device=device)

        if len(euc_dist.shape) <= 1:
            break
        else:
            selected_cluster_indices = torch.argmin(euc_dist, dim=1)

        center_inits = centers.clone()

        for index in range(num_clusters):
            selected_cluster = torch.nonzero(selected_cluster_indices == index).squeeze().to(device)
            chosen_indices = torch.index_select(X, 0, selected_cluster)

            if chosen_indices.shape[0] == 0:
                chosen_indices = X[torch.randint(len(X), (1,))]

            centers[index] = chosen_indices.mean(dim=0)

        # Calculate the delta from center_inits to centers
        center_delta_pow = torch.pow((centers - center_inits), 2)
        center_shift_pow = torch.pow(torch.sum(torch.sqrt(torch.sum(center_delta_pow, dim=1))), 2)

        # If the cluster centers are not changing significantly, stop the loop.
        if center_shift_pow < threshold:
            break

    return selected_cluster_indices, centers


def compute_adjacency_matrix(X: torch.tensor) -> torch.Tensor:
    # eq (6)
    # Compute cosine similarity matrix
    X_norm = F.normalize(X, p = 2, dim = 1)
    A = torch.mm(X_norm, X_norm.t())
    # Set diagonal to zero (no self-loops)
    A.fill_diagonal_(0)
    return A

def cluster_row(
        A: torch.Tensor, k: int = 2, device: torch.device = torch.device('cpu')
    ):
    # eq (7) -> find Ci_w, Ci_b

    # Using k-Means clustering to find two clusters
    _labels, centers = kmeans_torch(A.unsqueeze(1), num_clusters = k, device = device)

    # Identify Cw (cluster with the larger center)
    if centers[0] > centers[1]:
        C_w = (_labels == 0)
        C_b = (_labels == 1)
    else:
        C_w = (_labels == 1)
        C_b = (_labels == 0)

    return C_w, C_b

def retain_top_p(row, C_w, p):
    # Sort the values in C_w cluster and retain top p%
    C_w_values = row[C_w]
    threshold = torch.quantile(C_w_values, 1 - p / 100)
    P = torch.zeros_like(row)
    P[row >= threshold] = row[row >= threshold]
    return P

def compute_laplacian(P):
    # eq (9), (10)
    # Symmetric matrix
    W = (P + P.T) / 2  
    # Degree matrix
    D = torch.diag(torch.sum(W, dim=1))
    # Laplacian matrix
    L = D - W
    return L

def estimate_k(L, k_max):
    eigenvalues = eigvalsh(L)
    sorted_eigenvalues = torch.sort(eigenvalues)[0]
    
    # Compute eigengap
    eigengap = sorted_eigenvalues[1: ] - sorted_eigenvalues[: -1]
    eigengap = eigengap[: min(k_max, eigengap.shape[0])]
    
    # Estimate the number of clusters
    k_hat = torch.argmax(eigengap) + 1  # Adding 1 as index starts from 0
    return k_hat

def spectral_clustering(L, k_hat, device):
    L = L.float().to(device)
    _, eigenvectors = eigh(L)
    V = eigenvectors[:, : k_hat]  # Take the k_hat smallest eigenvectors
    
    # k-means on the spectral embeddings
    _labels, _ = kmeans_torch(V, num_clusters = k_hat, device = device)
    return _labels

def sc_pna(
        X,
        device = torch.device('cpu'),
        p = 20, 
        kmax = 8, 
    ):

    # Step 1: Create the adjacency matrix
    A = compute_adjacency_matrix(X)

    # Step 2, 3: Identify C_w and C_b and retain top p%
    P = torch.zeros_like(A)
    for i in range(A.shape[0]):
        C_w, C_b = cluster_row(A[i], k = 2, device = device)
        P[i] = retain_top_p(A[i], C_w, p)

    # Step 4: Symmetric matrix and Laplacian
    L = compute_laplacian(P)

    # Step 5: Estimate k using eigengap
    k_hat = estimate_k(L, kmax)

    # Step 6: Spectral clustering
    label = spectral_clustering(L, k_hat, device)

    return label

if __name__ == "__main__":
    X = torch.randn(100, 64)
    print(sc_pna(X))
