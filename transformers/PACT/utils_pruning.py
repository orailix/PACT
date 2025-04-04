
from abc import ABC, abstractmethod
import time
import math
import re
import time
import torch
import torch.nn as nn
import random
import numpy as np
import os
import torch
import time
import torch
import lmdb
import json
import faiss
from types import SimpleNamespace

def load_config(config_path=None):
    if config_path is None:
        config_path = os.getenv('pact_config_path', None)
    
    # Default values
    default_config = {  
        "visual_token_reduction": False,
        "layer_for_reduction": 4,
        "progessive_reduction": False,
        "use_DBDPC": False,
        "cutoff": 0.0,
        "vector_to_use_in_distance_clustering": "current_k_cosine",
        "take_mean": True,
        "include_pruned_in_mean": True,
        "do_not_consider_non_image_tokens_as_pruned": True,
        "coef_pruned": 1.5,
        "avoid_numerical_instability_DBDPC": True,
        "withdraw_visual_tokens": False,
        "VTW_equivalant_layer_for_reduction": -1,
        "equivalent_reduc_percentage_vtw": 0.0,
        "use_tome": False,
        "perc_tokeep_tome_total": 1.0,
        "tome_equivalant_layer_for_reduction": 4,
        "use_kmeans": False,
        "perc_tokeep_kmeans": 1.0,
        "use_dpc": False,
        "percentage_to_keep_dpc": 1.0,
        "use_agglomerative": False,
        "percentage_to_keep_agglomerative": 1.0,
        "linkage": "single",
        "use_dbscan": False,
        "eps_dbscan": 0.1,
        "noise_as_clusters_dbscan": False,
        "token_pruning": False,
        "use_all_non_text_pruning": True,
        "prune_with_norm": False,
        "use_cosine_in_token_pruning": True,
        "use_attention_in_token_pruning": False,
        "use_mask_in_use_attention_in_token_pruning": False,
        "use_IQR_in_token_pruning": False,
        "alpha_IQR": 0.5,
        "pruning_filter_wth_percentage": True,
        "pruning_tokeep_percentage_value": 1.0,
        "multiply_by_norm": False,
        "norm_to_use": 2,
        "avoid_numerical_instability_prune": True,
        "no_proportional_attention": False,
        "change_position_ids": False,
        "get_mean_position_id": False,
        "synchro": False,
        "need_kq": True,
        "do_not_upcast_to_full_precision_for_pruning": False,
        "keep_casual": True,
        "get_performance_metrics": False,
        "get_reduction_ratio": False,
        "use_custom_merging": False,
        "use_custom_pruning": False,
        "log_output_path" : "aggregated_metrics",
    }


    # Load configuration file if it exists
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_file = json.load(f)
    else:
        config_file = {}

    # Merge defaults with file config
    config = default_config.copy()
    config.update(config_file)

    # Override with any matching environment variables
    for key in config:
        env_val = os.getenv(key)
        if env_val is not None:
                default_type = type(default_config[key])
            # try:
                # Convert string env var to the type of the default
                if default_type == bool:
                    config[key] = env_val.lower() in ("true", "1", "yes")
                else:
                    config[key] = default_type(env_val)
            # except Exception:
            #     print(f"Warning: could not cast environment variable {key} to {default_type}, using default.")

    # Apply internal logic
    if config["synchro"]:
        config["get_performance_metrics"] = True
    if config["get_performance_metrics"]:
        config["synchro"] = True
    if config["use_tome"]:
        config["progessive_reduction"] = True
        config["layer_for_reduction"] = 0

     # --- Mutual Exclusivity Checks ---

    # 1. Only one of these reduction methods can be active
    exclusive_methods_1 = [
        "use_DBDPC", "use_tome", "use_kmeans",
        "use_dpc", "use_agglomerative", "use_dbscan"
    ]
    active_methods_1 = [key for key in exclusive_methods_1 if config.get(key)]
    if len(active_methods_1) > 1:
        raise ValueError(
            f"Only one of the following can be true at a time: {', '.join(exclusive_methods_1)}. "
            f"Currently enabled: {', '.join(active_methods_1)}"
        )

    # 2. Only one of these pruning methods can be active
    exclusive_methods_2 = [
        "prune_with_norm", "token_pruning", "withdraw_visual_tokens"
    ]
    active_methods_2 = [key for key in exclusive_methods_2 if config.get(key)]
    if len(active_methods_2) > 1:
        raise ValueError(
            f"Only one of the following can be true at a time: {', '.join(exclusive_methods_2)}. "
            f"Currently enabled: {', '.join(active_methods_2)}"
        )

    # 3. If token_pruning is true, exactly one of use_IQR_in_token_pruning or pruning_filter_wth_percentage must be true
    if config.get("token_pruning"):
        pruning_options = [
            config.get("use_IQR_in_token_pruning", False),
            config.get("pruning_filter_wth_percentage", False)
        ]
        if pruning_options.count(True) != 1:
            raise ValueError(
                "When 'token_pruning' is enabled, exactly one of "
                "'use_IQR_in_token_pruning' or 'pruning_filter_wth_percentage' must be set to True."
            )

    return SimpleNamespace(**config)

def normalize(tensor):
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)  # To avoid division by zero
    return tensor / norm


def normal_compute_pairwise_distances(X,l_2=False):
  
    X = torch.nn.functional.normalize(X,p=2.0,)
    
    dot_product = torch.mm(X, X.t())
 
    cosine_distance = 1 - dot_product
    
    return cosine_distance



class DBDPC:
    def __init__(self, dc=2):
        self.dc = dc
        self.treated_dimensions={}
    
    def get_clusters(self):
        if not hasattr(self, 'labels_') or self.labels_ is None:
            raise ValueError("Labels have not been initialized. Run the clustering algorithm first.")
        
        clusters = {label: np.where(self.labels_ == label)[0] for label in np.unique(self.labels_) if label!=-1}
        
        return clusters


    def fit_variant(self, X, cutoff,pact_config, pruned_keys=None, print_=False):
        """
        Clustering function that recursively identifies cluster centers and assigns labels to data points.
        If in any iteration fewer than 10 new centers are identified, it reverts to the previous method.

        Args:
            X: Data matrix of shape [N, D], where N is the number of points and D is the number of features.
            cutoff: Cutoff value for assigning labels.
            pruned_keys: Optional tensor of pruned keys to include in the assignment.
            print_: If True, prints timing and debug information.

        Returns:
            labels: Tensor of cluster labels for each data point.
        """
        device = X.device
        N = X.shape[0]
        cluster_centers = []
        unassigned_mask = torch.ones(N, dtype=torch.bool, device=device)  # True for unassigned points

        print_= print_ and pact_config.synchro
        # Compute the initial pairwise distance matrix
        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            start = time.time()

        dist = torch.clamp(normal_compute_pairwise_distances(X), min=0)
        dist.fill_diagonal_(0)

        if pact_config.synchro and print_ :
            torch.cuda.synchronize()
            if print_:
                print(f"Distance matrix computation takes {time.time() - start} seconds")

        if print_:
            print(f"dist max is {dist.max().item()}")

        if pact_config.synchro and print_ :
            torch.cuda.synchronize()
            start = time.time()
        # print(f"cutoff_density {cutoff_density} restrict_density {restrict_density}")
        # if cutoff_density :
        #     print("cutoff density")
        #     rho = (torch.exp(-(dist /cutoff)**2)).sum(dim=1)  # Sum over all points
        # elif restrict_density:
        #     print("restricting density")
        #     rho = (torch.exp(-(dist / self.dc)**2) * (dist < cutoff)).sum(dim=1) 
        # else :
        rho = torch.exp(-(dist / self.dc)**2).sum(dim=1)  # Sum over all points

        
        if pact_config.avoid_numerical_instability_DBDPC :
            sorted_indices = torch.argsort(rho)
            ranks = torch.empty_like(sorted_indices).to(rho.device).to(rho.dtype)
            ranks[sorted_indices] = torch.arange(len(rho)).to(rho.device).to(rho.dtype)
            rho=ranks



        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            if print_:
                print(f"Rho calculations take {time.time() - start} seconds")

        iteration = 0
        if pact_config.synchro and print_:
            torch.cuda.synchronize()
            start = time.time()

        while True:
            iteration += 1
            if print_:
                print(f"\nIteration {iteration}: Starting with {unassigned_mask.sum().item()} unassigned points.")

            unassigned_indices = torch.where(unassigned_mask)[0]
            num_unassigned = unassigned_indices.size(0)
            if num_unassigned == 0:
                break  # No unassigned points left

            # Extract distances and rho for unassigned points
            dist_unassigned = dist[unassigned_indices][:, unassigned_indices]
            rho_unassigned = rho[unassigned_indices]

            # Calculate delta for unassigned points
            rho_expand = rho_unassigned.unsqueeze(1).expand(num_unassigned, num_unassigned)


            rho_compare = ~torch.gt(rho_expand, rho_expand.t())
            rho_compare.fill_diagonal_(False)

            inf_mask = torch.full_like(dist_unassigned, float('inf'), dtype=dist_unassigned.dtype)
            conditioned_spatial_dist_matrix = torch.where(rho_compare, dist_unassigned, inf_mask)

            delta_unassigned, _ = torch.min(conditioned_spatial_dist_matrix, dim=1)

            # Handle the point(s) with the highest density
            max_rho = rho_unassigned.max()
            max_rho_mask = (rho_unassigned == max_rho)
            delta_unassigned[max_rho_mask] = float('inf')  # Keep delta as infinity for maximum rho points

            # Identify new cluster centers where delta > cutoff
            new_centers_mask = delta_unassigned > cutoff
            new_centers_indices = unassigned_indices[new_centers_mask]

            num_new_centers = new_centers_indices.numel()
            if num_new_centers == 0:
                if print_:
                    print("No new cluster centers found.")
                break  # No new cluster centers found

            # Add new cluster centers
            cluster_centers.extend(new_centers_indices.tolist())

            if print_:
                print(f"Identified {num_new_centers} new cluster centers.")

            # Exclude points within cutoff of new cluster centers
            dist_to_new_centers = dist[unassigned_indices][:, new_centers_indices]
            within_cutoff = (dist_to_new_centers <= cutoff).any(dim=1)

            # Update unassigned_mask: Mark points within cutoff as assigned
            points_within_cutoff = unassigned_indices[within_cutoff]
            unassigned_mask[points_within_cutoff] = False

            # Also mark the new cluster centers as assigned
            unassigned_mask[new_centers_indices] = False

            # If fewer than 10 new centers are identified, revert to the previous method
            if num_new_centers < 10:
                if print_:
                    print("Fewer than 10 new centers identified, reverting to previous method.")

                # Prepare variables needed
                delta = delta_unassigned.clone()
                # Sort indices based on delta in descending order
                delta_sorted_indices = unassigned_indices[torch.argsort(-delta_unassigned)]
                delta_sorted_values = delta_unassigned[torch.argsort(-delta_unassigned)]

                if pact_config.synchro and print_ :
                    torch.cuda.synchronize()
                    start = time.time()

                first_index = torch.searchsorted(-delta_sorted_values, -cutoff)

                # Keep everything in tensor form for faster execution
                cluster_centers.extend(delta_sorted_indices[:first_index].tolist())

                # Update delta_sorted_indices to exclude the cluster centers
                delta_sorted_indices = delta_sorted_indices[first_index:]

                rho_unassigned_sorted = rho[delta_sorted_indices]
                delta_sorted_indices = delta_sorted_indices[torch.argsort(-rho_unassigned_sorted)]

                # Calculate the distances for non-cluster centers
                dist_to_centers = dist[delta_sorted_indices][:, cluster_centers]

                # Check which points are not within the target distance to any cluster center
                not_within_cutoff_all = (dist_to_centers > cutoff).all(dim=1)

                # Filter out the delta indices that are not within the distance criterion
                delta_sorted_indices = delta_sorted_indices[not_within_cutoff_all]

                # Compute the pairwise distances between the remaining delta_sorted_indices
                dist_mapped = dist[delta_sorted_indices][:, delta_sorted_indices]

                # Create a mask for distances within cutoff
                within_cutoff = (dist_mapped <= cutoff)
                within_cutoff.fill_diagonal_(False)

                # Create an assigned mask with zeros
                assigned_mask = torch.zeros(dist_mapped.shape[0], dtype=torch.bool, device=dist.device)

                # Check if any points are within the target distance
                within_cutoff_any = within_cutoff.any(dim=0)

                # Use the 'for' loop from the previous method
                for index, i in enumerate(delta_sorted_indices):
                    if not assigned_mask[index]:
                        cluster_centers.append(i.item())
                        if within_cutoff_any[index]:
                            assigned_mask[index + 1:] |= within_cutoff[index, index + 1:]

           

                break  # Exit the while loop after reverting to previous method

        if pact_config.synchro and print_ :
            torch.cuda.synchronize()
            if print_:
                print(f"New center recursive identification calculations take {time.time() - start} seconds")

        cluster_centers = torch.tensor(cluster_centers, device=device)

        # Assign labels to data points
        if pact_config.synchro and print_ :
            torch.cuda.synchronize()
            start = time.time()

        if pruned_keys is None:
            # Assign labels based on nearest cluster centers
            dist_to_centers = dist[:, cluster_centers]  # Shape: [N, C]
            nearest_center = torch.argmin(dist_to_centers, dim=1)  # Shape: [N]
            labels = cluster_centers[nearest_center]  # Shape: [N]
            labels[cluster_centers] = cluster_centers  # Ensure cluster centers are labeled correctly
        else:
            
            # use_mean_instead_for_retreival=True
            # if use_mean_instead_for_retreival:
            #     # Additional processing if pruned_keys are provided
            #     X_normalized = torch.nn.functional.normalize(X, p=2.0, dim=1)  # Shape: [N, D]
            #     pruned_keys_normalized = torch.nn.functional.normalize(pruned_keys, p=2.0, dim=1)  # Shape: [M, D]

            #     num_initial = X_normalized.size(0)  # N
            #     num_pruned = pruned_keys_normalized.size(0)  # M
            #     num_tokens = num_initial + num_pruned  # N + M

            #     # Calculate labels normally for non-pruned points (DO NOT CHANGE THESE LABELS)
            #     dist_to_centers = dist[:, cluster_centers]  # Shape: [N, C]
            #     nearest_center = torch.argmin(dist_to_centers, dim=1)  # Shape: [N]
            #     labels_initial = cluster_centers[nearest_center]  # Shape: [N]
            #     labels_initial[cluster_centers] = cluster_centers  # Ensure cluster centers are labeled correctly

            #     # Calculate mean vectors for retrieval instead of original centers (ADDED)
            #     clusters = {center.item(): (labels_initial == center).nonzero(as_tuple=True)[0].tolist() for center in cluster_centers}
            #     cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(X, clusters)
            #     cluster_center_vectors = torch.nn.functional.normalize(cluster_means, p=2.0, dim=1)

            #     # Compute distances ONLY for pruned tokens (optimized, no redundant computation for non-pruned tokens)
            #     similarity_pruned = torch.matmul(torch.cat((X_normalized,pruned_keys_normalized),dim=0), cluster_center_vectors.transpose(0, 1))  # Shape: [M, C]
            #     dist_pruned = 1.0 - similarity_pruned  # Shape: [M, C]
            #     min_distances_pruned, nearest_center_indices_pruned = torch.min(dist_pruned, dim=1)  # Shape: [M]

            #     cutoff = cutoff * pact_config.coef_pruned
            #     labels = torch.full((num_tokens,), -1, dtype=torch.long, device=device)  # Shape: [N + M]

            #     # Set labels directly from initial calculation (unpruned points)
            #     # labels[:num_initial] = labels_initial  # (ADDED) directly use labels_initial here

            #     # Set labels for pruned points based on distances (ADDED clearly separated)
            #     mask_pruned = min_distances_pruned <= cutoff  # Shape: [M]
            #     # labels[num_initial:][mask_pruned] = cluster_centers[nearest_center_indices_pruned[mask_pruned]]
            #     labels[mask_pruned] = cluster_centers[nearest_center_indices_pruned[mask_pruned]]
                


            # else:
            # Additional processing if pruned_keys are provided
            X_normalized = torch.nn.functional.normalize(X, p=2.0, dim=1)  # Shape: [N, D]
            pruned_keys_normalized = torch.nn.functional.normalize(pruned_keys, p=2.0, dim=1)  # Shape: [M, D]

            # Concatenate tokens
            all_tokens = torch.cat((X_normalized, pruned_keys_normalized), dim=0)  # Shape: [N + M, D]
            num_initial = X_normalized.size(0)  # N
            num_pruned = pruned_keys_normalized.size(0)  # M
            num_tokens = num_initial + num_pruned  # N + M

            # Retrieve cluster center vectors
            cluster_center_vectors = X_normalized[cluster_centers]  # Shape: [C, D]

            # Compute distances
            # A. For X Tokens
            dist_to_centers_xuntouched = dist[:, cluster_centers]  # Shape: [N, C]

            # B. For pruned_keys Tokens
            similarity_pruned = torch.matmul(pruned_keys_normalized, cluster_center_vectors.transpose(0, 1))  # Shape: [M, C]
            dist_pruned = 1.0 - similarity_pruned  # Shape: [M, C]

            # Concatenate distances
            all_dist_to_centers = torch.cat((dist_to_centers_xuntouched, dist_pruned), dim=0)  # Shape: [N + M, C]

            # Find the nearest cluster center
            min_distances, nearest_center_indices = torch.min(all_dist_to_centers, dim=1)  # Shape: [N + M]

            cutoff = cutoff * pact_config.coef_pruned

            labels = torch.full((num_tokens,), -1, dtype=torch.long, device=device)  # Shape: [N + M]
            mask = min_distances <= cutoff  # Shape: [N + M]
            labels[mask] = cluster_centers[nearest_center_indices[mask]]
            labels[cluster_centers] = cluster_centers  # Ensure cluster centers are labeled correctly

            ##add here weight_redistribution which for element that do not respect  min_distances <= cutoff calculate a weight distribution proportional to 1/dist from each point 

        if pact_config.synchro and print_ :
            torch.cuda.synchronize()
            if print_:
                print(f"Point assignment to nearest cluster centers takes {time.time() - start} seconds")

        self.labels_ = labels.cpu().numpy()
        

def compute_fastest_cluster_means_with_arbitrary_ids(tensor, clusters, weights_list=None, synchro=False,print_=False):
    """
    Compute the mean of elements in each cluster using scatter_add, assuming all clusters have at least one element.
    
    Args:
        tensor: A tensor where each row represents an item, and columns are features (shape [N, D]).
        clusters: A dictionary where each key is a cluster_id and the value is a list of indices belonging to that cluster.
        weights_list: An optional tensor or list representing the number of elements in each cluster.
                      If not provided, the function computes the number of elements per cluster.
        synchro: If True, synchronizes CUDA operations and measures preparation time.
    
    Returns:
        cluster_means: A tensor where each row is the mean of the elements in the corresponding cluster.
    """
    
    if synchro and print_:
        torch.cuda.synchronize()
        start = time.time()

    # Map cluster IDs to continuous indices
    cluster_id_list = list(clusters.keys())
    id_to_index = {cluster_id: i for i, cluster_id in enumerate(cluster_id_list)}

    num_clusters = len(cluster_id_list)
    feature_dim = tensor.size(1)
 
    # Preallocate storage for sum accumulation
    cluster_sums = torch.zeros((num_clusters, feature_dim), device=tensor.device, dtype=tensor.dtype)


    all_indices_list = []
    
    # Populate the list with indices from each cluster
    for indices in clusters.values():
        all_indices_list.extend(indices)  # Extend the list with indices directly
    
    # Convert the entire list to a tensor in one operation
    all_indices = torch.tensor(all_indices_list, dtype=torch.long, device=tensor.device)

    cluster_indices_list = []
    
    # Populate the list with the continuous index for each cluster
    for cluster_id, indices in clusters.items():
        cluster_indices_list.extend([id_to_index[cluster_id]] * len(indices))  # Add the index len(indices) times
    
    # Convert the entire list to a tensor in one operation
    cluster_indices = torch.tensor(cluster_indices_list, dtype=torch.long, device=tensor.device)
    
    # Compute cluster sizes if weights_list is not provided
    if weights_list is None:
        cluster_sizes = torch.zeros(num_clusters, device=tensor.device, dtype=tensor.dtype)
        cluster_sizes.index_add_(0, cluster_indices, torch.ones_like(cluster_indices, dtype=tensor.dtype))
    else:
        if isinstance(weights_list, list):
            weights_list = torch.tensor(weights_list, dtype=tensor.dtype, device=tensor.device)
        cluster_sizes = weights_list

    if synchro and print_:
        torch.cuda.synchronize()
        print(f"Preparation for merging took {time.time() - start}")

    # Compute the sum of values in each cluster
    cluster_sums.index_add_(0, cluster_indices, tensor[all_indices])

    
    # Compute the cluster means directly
    cluster_means = cluster_sums / cluster_sizes.unsqueeze(1)

    return cluster_means

def merge_clusters(tensor, clusters,pact_config,position_ids=None,pruned_hiddens=None,print_=False ):
    """
    Merges clusters in the given tensor based on the provided clusters dictionary.
    If base_keys, pruned_hiddens, pruned_keys, and cutoff are provided, it further decides whether
    additional points in pruned_hiddens need to be merged into existing clusters based on
    the distance defined as 1 - cosine_similarity using the keys.

    Args:
        tensor: The original tensor containing all items.
        clusters: A dictionary where each key is a cluster_id and the value is a list of indices belonging to that cluster.
        base_keys: The keys corresponding to the tensor, used to compute cosine similarity.
        pruned_hiddens: Additional points that may need to be merged into clusters.
        pruned_keys: The keys corresponding to pruned_hiddens.
        cutoff: The maximum allowed distance for merging a pruned point into a cluster.
        Other arguments remain the same.

    Returns:
        Depending on the arguments, returns the merged tensor, mask, weights, and position_ids.
    """
    if pact_config.synchro and print_ :
        torch.cuda.synchronize()
    start = time.time()
    # sorted_clusters = dict(sorted(clusters.items()))
    cluster_indices = [cluster_id for cluster_id, indices in clusters.items()]
    weights_list = [len(indices) for cluster_id, indices in clusters.items()]
    # position_ids_list = [int_toward_base(sum(indices) / len(indices), cluster_id) for cluster_id, indices in clusters.items()]
    start_2 = time.time()
   
    if pruned_hiddens is not None :
        tensor_with_pruned = torch.cat([tensor, pruned_hiddens], dim=0)
        if pact_config.take_mean:
            cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(tensor_with_pruned,clusters,weights_list,synchro=pact_config.synchro)
        else :
            cluster_means = torch.stack([tensor_with_pruned[cluster_id] for cluster_id, indices in clusters.items()])
    else :
        if pact_config.take_mean:
            cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(tensor, clusters,weights_list,synchro=pact_config.synchro)
        else:
            cluster_means = torch.stack([tensor[cluster_id] for cluster_id, indices in clusters.items()])
    
    if pact_config.synchro and print_ :
        torch.cuda.synchronize()
        if print_ :
            print(f"Merging calculation took {time.time() - start}")


    # Prepare output_tensor, mask, weights, position_ids
    output_tensor = torch.zeros_like(tensor, dtype=tensor.dtype, device=tensor.device)
    mask = torch.zeros((tensor.size(0), 1), dtype=torch.bool, device=tensor.device)

    # Map cluster IDs to indices
    # cluster_id_list = list(clusters.keys())
   
    weights = torch.zeros_like(tensor[:, 0], dtype=torch.float32, device=tensor.device)
    weights_centers = torch.tensor(weights_list, dtype=torch.float32, device=tensor.device)
    weights[cluster_indices] = weights_centers
  
    # Update output_tensor, mask, and positions in a vectorized way
    output_tensor[cluster_indices] = cluster_means
    mask[cluster_indices, 0] = True
    if pact_config.synchro and print_ :
        if print_ :
            print(f"Got {mask.sum()} centers")
    mask = mask.to(torch.bool)

    if pact_config.synchro and print_ :
        torch.cuda.synchronize()
        if print_ :
            print(f"Merging calculation took {time.time() - start}")
    if pact_config.get_mean_position_id:
        # position_ids shape: (1, N)
        # clusters: dict[int, list[int]], where values are token indices (along N)
        position_ids_output = torch.zeros_like(position_ids)  # shape (1, N)

        for cluster_id, indices in clusters.items():
            # import pdb; pdb.set_trace()

            values = position_ids[indices]  # slice tensor directly
            mean_val = torch.round(values.float().mean()).long()
            position_ids_output[indices] = mean_val

        return output_tensor, mask, weights, position_ids_output
    else:
        return output_tensor, mask, weights, None


def token_reduction(image_feature,image_feature_for_clustering,cutoff,reduction,pact_config,position_ids=None,pruned_hiddens=None,pruned_keys=None) :
           
    dbdpc_variant=DBDPC(dc=2) 
    
    dbdpc_variant.fit_variant(image_feature_for_clustering,cutoff,pact_config=pact_config,pruned_keys=pruned_keys)
  
    clusters_variant = dbdpc_variant.get_clusters()

    merged,mask_image,weights,position_ids_output = merge_clusters(image_feature, clusters_variant,position_ids=position_ids, pruned_hiddens=pruned_hiddens,pact_config=pact_config)
    image_feature=merged

    
    # if pact_config.synchro :
    #     reduction[1]+=mask_image.sum().item()
        
    return  image_feature,mask_image,weights,position_ids_output

def custom_token_reduction(image_feature, image_feature_for_reduction, position_ids=None, pruned_hiddens=None, pruned_for_reduction=None, cutoff=0):
    """
    Performs token reduction on image features.

    Inputs:
    - image_feature (Tensor, shape N, D):  
      The hidden states to be reduced.
      
    - image_feature_for_reduction (Tensor, shape N, D'):  
      Vectors used in the token reduction process.
      The specific vectors used can be configured via config.vector_to_use_in_distance_clustering.  
      Default: Keys after the application of rotary embeddings.

    - position_ids (Tensor, shape N) or (Tensor, shape (N,1,3)) for Qwen2-VL:  
      The position IDs.

    - pruned_hiddens (Tensor, shape N, D, optional):  
      If config.include_pruned_in_mean = True, this contains the hidden states of the pruned tokens, if a pruning step was done.

    - pruned_for_reduction (Tensor, shape N, D', optional):  
      If config.include_pruned_in_mean = True, this contains the same type of vectors as image_feature_for_reduction
      but for pruned tokens, so defalut is the Keys after the application of rotary embeddings of pruned vectors.

    - cutoff (float):  
      A hyperparameter (defined in the config file) that typically controls the reduction intensity.

    Outputs:
    - image_feature (Tensor, shape N, D):  
      The updated image features after token reduction (same shape as input).

    - position_ids (Tensor, shape N) or (Tensor, shape (N,3,1)) for Qwen2-VL:  
      The uptaded position IDs (if modified during reduction).

    - mask_image (Tensor, shape N):  
      A binary mask indicating which tokens are kept (1) vs. removed (0).

    - weights (Tensor, shape N):  
      Weights used for applying proportional attention.  
      If proportional attention is disabled, this returns a vector of ones (default behavior).  
      This behavior can be adjusted in the config.

    Notes:
    - If tokens are merged, one token must be selected as the representative, nd it's value should be repalced by  
      the merged tokens resulting vector.  
    - The mask should mark this selected token as 1, while all other merged tokens should be 0.
    - Only the following will be used in further processing:  
      image_feature[:, mask_image], weights[:, mask_image], position_ids_output[:, mask_image].
    """

    return image_feature, mask_image, weights, position_ids

def custom_pruning(k_image, q_image):
    """
    Computes token importance scores for pruning based on the similarity between image keys and queries.

    Inputs:
    - k_image (Tensor): 
        Shape: (B, H_kq, N_k, D)
        The key tensor.
        B = batch size  
        H_kq = number of attention heads
        N_k = keys sequence length
        D = head dimension
    
    - q_image (Tensor): 
        the query tensor
        Shape: (B, H_kq, N_q, D)
        The processed image query tensor.  
        B = batch size  
        H_q =  number of attention heads
        N_q = query sequence length  
        D = head dimension

    Output:
    - scores (Tensor): 
        Shape: (B, N_kv)
        Importance scores for each image key token, aggregated across heads and query positions.
        These scores will be used to prune the visual tokens.

    Notes:
    - If `pact_config.use_cosine_in_token_pruning` is set to True, rotary embeddings are applied to both keys and queries.
    - The number of tokens N_k and N_q can differ.
      They are equal when `pact_config.use_all_non_text_pruning` is set to True, in which case they represent all non-textual tokens.
      Otherwise, N_k typically refers only to the visual tokens used for pruning.
    - Although pruning is applied only to visual tokens, including special tokens in the key tensor (K) can still be beneficial.
      For example, when using softmax-based scoring, special tokens may provide additional context that influences attention distribution.
    """

    return scores

def bipartite_soft_matching(metric, r):
    """
    Performs bipartite soft matching and finds connected components in the bipartite graph,
    assigning labels based on connections between A and B.

    Args:
        metric (torch.Tensor): A similarity matrix of shape [num_tokens, feature_dim].
        r (int): The number of most similar edges to keep.

    Returns:
        labels (torch.Tensor): A tensor of shape [num_tokens] where labels[i] indicates the component id of node i.
    """
    num_tokens = metric.shape[0]
    device = metric.device

    # Partition the tokens into two sets A and B (even and odd indices)
    A_indices = torch.arange(0, num_tokens, 2, device=device)
    B_indices = torch.arange(1, num_tokens, 2, device=device)

    # Compute similarity between tokens in A and B
    A_to_B_sim = torch.matmul(metric[A_indices], metric[B_indices].T)  # Similarity between A and B

    # For each token in A, find the most similar token in B
    most_similar_B = A_to_B_sim.argmax(dim=-1)  # Indices of the most similar tokens in B for each A

    # Get the similarity values for the most similar pairs
    similarity_values = A_to_B_sim.max(dim=-1)[0]

    # Select the r most similar edges (pairs)
    top_r_similarities, top_r_indices = torch.topk(similarity_values, r)

    # Indices of tokens in A and B for the top r pairs
    A_top_r = A_indices[top_r_indices]
    B_top_r = B_indices[most_similar_B[top_r_indices]]

    # Initialize labels as their own indices
    labels = torch.arange(num_tokens, device=device)

    # Assign labels: Any point in A connected to a point in B gets the label of the point in B
    labels[A_top_r] = labels[B_top_r]

    # Other points' labels remain their own indices
    return labels

def merge_connected_components(X, sizes, labels):
    """
    Merge tokens within each connected component using the labels,
    mapping them back to the full sequence length without using first_occurrence_indices.

    Args:
        X (torch.Tensor): Input tensor of tokens to merge (shape [num_tokens, dim]).
        sizes (torch.Tensor): Tensor of token sizes (weights) with shape [num_tokens].
        labels (torch.Tensor): Tensor of component labels with shape [num_tokens].

    Returns:
        new_X (torch.Tensor): Merged tensor of tokens.
        new_sizes (torch.Tensor): Updated sizes (weights) of the merged tokens.
        mask (torch.Tensor): Mask indicating which tokens were kept.
    """
    device = X.device
    num_tokens, dim = X.shape

    # Get unique labels and inverse indices to map labels to components
    unique_labels, inverse_indices = torch.unique(labels, sorted=True, return_inverse=True)
    num_components = unique_labels.size(0)

    # Compute per-component total sizes
    component_sizes = torch.zeros(num_components, device=device).to(sizes.dtype).scatter_add(
        0, inverse_indices, sizes
    )

    # Compute per-component weighted sums
    weighted_X = X * sizes.unsqueeze(-1)
    component_weighted_sum = torch.zeros(num_components, dim, device=device).to(weighted_X.dtype).scatter_add(
        0, inverse_indices.unsqueeze(-1).expand(-1, dim), weighted_X
    )

    # Compute per-component merged tokens
    merged_tokens = component_weighted_sum / component_sizes.unsqueeze(-1).to(component_weighted_sum.dtype)


    # Prepare new_X and new_sizes
    new_X = torch.zeros_like(X)
    new_sizes = torch.zeros_like(sizes)
    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)

    # Map merged tokens back to the full sequence length using labels
    new_X[unique_labels] = merged_tokens.to(new_X.dtype)
    new_sizes[unique_labels] = component_sizes
    mask[unique_labels] = True

    return new_X, new_sizes, mask

class TOME:
    def __init__(self, r):
        """
        Args:
            r (int): The number of most similar edges to keep for merging.
        """
        self.r = r

    def fit(self, X_clustering, X_merge, sizes):
        """
        Applies TOME token merging by finding connected components based on the r most similar edges.

        Args:
            X_clustering (torch.Tensor): Input tensor used to calculate similarity (shape [num_tokens, feature_dim]).
            X_merge (torch.Tensor): Input tensor of tokens that will be merged (shape [num_tokens, feature_dim]).
            sizes (torch.Tensor): Tensor containing the sizes (weights) of tokens (shape [num_tokens]).

        Returns:
            new_X (torch.Tensor): Tensor with the merged tokens.
            mask (torch.Tensor): Mask indicating which tokens were kept (1) and which were discarded (0).
            new_sizes (torch.Tensor): Updated sizes (weights) of the tokens.
        """
        num_tokens = X_clustering.shape[0]

        # Ensure r is valid
        self.r = max(min(self.r, num_tokens // 2 - 2), 0)

        # Normalize the token representations for similarity calculation
        X_clustering_norm = X_clustering / X_clustering.norm(dim=-1, keepdim=True)

        # Build the bipartite graph and assign labels based on connections
        labels = bipartite_soft_matching(X_clustering_norm, self.r)

        # Merge tokens within each connected component using the labels
        merged_X, new_sizes, mask = merge_connected_components(X_merge, sizes, labels)

        return merged_X, mask, new_sizes

def token_reduction_tome(image_feature, image_feature_for_clustering, sizes, reduction, r=10):
    """
    Uses TOME to reduce the number of tokens by merging connected tokens in `image_feature`
    based on similarity calculated from `image_feature_for_clustering`.

    Args:
        image_feature (torch.Tensor): Input tensor of tokens to merge (shape [num_tokens, dim]).
        image_feature_for_clustering (torch.Tensor): Input tensor used to calculate similarity (shape [num_tokens, dim]).
        sizes (torch.Tensor): Tensor containing the sizes (weights) of tokens (shape [num_tokens]).
        reduction (list): A list to track the reduction ratio.
        r (int): Number of most similar edges to keep for merging.

    Returns:
        merged (torch.Tensor): The merged tensor after TOME.
        mask (torch.Tensor): Binary mask indicating which tokens were kept.
        weights (torch.Tensor): The updated sizes/weights of the tokens after merging.
    """
    tome_model = TOME(r=r)

    # Apply TOME using `image_feature_for_clustering` for similarity and `image_feature` for merging
    merged, mask_image, weights = tome_model.fit(
        image_feature_for_clustering, image_feature, sizes
    )


    # Update reduction statistics
    # reduction[1] += mask_image.sum().item()
    
    return merged, mask_image, weights, None



def token_reduction_kmeans(X_merge, X_clustering, k, max_iters=100):
    """
    Performs K-Means clustering using FAISS on GPU and updates the features in X_merge
    such that only the center of each cluster gets the mean of the features.

    Args:
        X_merge (torch.Tensor): Input data for merging of shape [n_samples, n_features_merge].
        X_clustering (torch.Tensor): Input data for clustering of shape [n_samples, n_features_clustering].
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.

    Returns:
        merged (torch.Tensor): Tensor of shape [n_samples, n_features_merge], where only the center
                               of each cluster is updated to the mean of the cluster in X_merge.
        mask (torch.Tensor): Binary mask of shape [n_samples] indicating if a point is the closest to its cluster center (1 for center, 0 otherwise).
        weights (torch.Tensor): Tensor of shape [n_samples], where only the center of each cluster has the cluster size.
        None: Placeholder for additional output.
    """
    n_samples, n_features_merge = X_merge.shape
    _, n_features_clustering = X_clustering.shape

    # Move data to CPU for FAISS compatibility
    X_clustering_np = X_clustering.cpu().numpy()

    # Step 1: Perform K-Means clustering using FAISS on GPU
    kmeans = faiss.Kmeans(d=n_features_clustering, k=k, niter=max_iters, gpu=True)
    kmeans.train(X_clustering_np)
    labels = kmeans.index.search(X_clustering_np, 1)[1].flatten()  # Cluster assignments
    centroids = torch.tensor(kmeans.centroids).to(X_merge.device)

    # Convert labels back to torch tensor on GPU
    labels = torch.tensor(labels, device=X_merge.device)

    # Step 2: Initialize merged output, mask, and weights for X_merge
    merged = torch.zeros_like(X_merge)  # Only the center will get the mean from X_merge
    mask = torch.zeros(n_samples, dtype=torch.bool, device=X_merge.device) # Only the center will get 1
    weights = torch.zeros(n_samples, device=X_merge.device)  # Only the center will get the cluster size

    # Step 3: For each cluster, find the point in X_clustering that is closest to the centroid
    for i in range(k):
        # Points in the i-th cluster
        cluster_points_clustering = X_clustering[labels == i]
        cluster_points_merge = X_merge[labels == i]  # Points in X_merge for this cluster
        cluster_indices = torch.where(labels == i)[0]  # Indices of the points in the cluster
        
        if cluster_points_clustering.shape[0] > 0:
            # Find the closest point to the centroid in X_clustering
            centroid = centroids[i].unsqueeze(0)  # Shape [1, n_features_clustering]
            distances = torch.cdist(cluster_points_clustering.to(torch.float32), centroid.to(torch.float32))  # Shape [n_cluster_points, 1]
            closest_idx = distances.argmin().item()

            # Get the index of the closest point in the original data
            closest_point_idx = cluster_indices[closest_idx]

            # Set the features of the closest point in X_merge to the mean of the cluster in X_merge
            merged[closest_point_idx] = cluster_points_merge.mean(dim=0)

            # Set the mask to 1 for the closest point (cluster center)
            mask[closest_point_idx] = 1

            # Set the weight of the closest point to the number of points in the cluster
            weights[closest_point_idx] = cluster_points_clustering.shape[0]

    return merged, mask, weights, None


def token_reduction_kmeans_base(X_merge, X_clustering, k, max_iters=100, tol=1e-4, device='cuda'):
    """
    Performs K-Means clustering using cosine similarity based on X_clustering features and updates the features 
    in X_merge such that only the center of each cluster gets the mean of the features.

    Args:
        X_merge (torch.Tensor): Input data for merging of shape [n_samples, n_features_merge].
        X_clustering (torch.Tensor): Input data for clustering of shape [n_samples, n_features_clustering].
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid convergence.
        device (str): Device to run on ('cuda' for GPU, 'cpu' for CPU).

    Returns:
        merged (torch.Tensor): Tensor of shape [n_samples, n_features_merge], where only the center
                               of each cluster is updated to the mean of the cluster in X_merge.
        mask (torch.Tensor): Binary mask of shape [n_samples] indicating if a point is the closest to its cluster center (1 for center, 0 otherwise).
        weights (torch.Tensor): Tensor of shape [n_samples], where only the center of each cluster has the cluster size.
        None: Placeholder for additional output.
    """
    # Move input tensors to the specified device (GPU or CPU)
    X_merge = X_merge.to(device)
    X_clustering = X_clustering.to(device)
    
    n_samples, n_features_merge = X_merge.shape
    _, n_features_clustering = X_clustering.shape
    
    # Normalize X_clustering to make it suitable for cosine similarity
    X_clustering_normalized = X_clustering / X_clustering.norm(dim=1, keepdim=True)

    # Step 1: Initialize random centroids from X_clustering
    centroids = X_clustering_normalized[torch.randperm(n_samples)[:k]].to(device)

    for _ in range(max_iters):
        # Step 2: Compute cosine similarity (dot product) between all points and centroids
        similarities = torch.matmul(X_clustering_normalized, centroids.T)  # [n_samples, k]

        # Assign points to the closest centroid (highest cosine similarity)
        labels = similarities.argmax(dim=1)

        # Step 3: Update centroids by taking the mean of the assigned points
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            cluster_points = X_clustering_normalized[labels == i]
            if cluster_points.shape[0] > 0:
                new_centroids[i] = cluster_points.mean(dim=0)
            else:
                # Handle empty clusters by reinitializing the centroid randomly
                new_centroids[i] = X_clustering_normalized[torch.randint(0, n_samples, (1,))]

        # Step 4: Check for convergence (if centroids don't move much)
        centroid_shift = torch.norm(new_centroids - centroids, dim=1).sum().item()
        centroids = new_centroids
        if centroid_shift < tol:
            break

    # Step 5: Initialize merged output, mask, and weights for X_merge
    merged = torch.zeros_like(X_merge, device=device)  # Only the center will get the mean from X_merge
    mask = torch.zeros(n_samples, dtype=torch.bool, device=device)  # Only the center will get 1
    weights = torch.zeros(n_samples, device=device)  # Only the center will get the cluster size

    # Step 6: For each cluster, find the point in X_clustering that is closest to the centroid
    for i in range(k):
        # Points in the i-th cluster
        cluster_points_clustering = X_clustering[labels == i]
        cluster_points_merge = X_merge[labels == i]  # Points in X_merge for this cluster
        cluster_indices = torch.where(labels == i)[0]  # Indices of the points in the cluster
        
        if cluster_points_clustering.shape[0] > 0:
            # Find the closest point to the centroid in X_clustering
            centroid = centroids[i].unsqueeze(0)  # Shape [1, n_features_clustering]
            distances = torch.cdist(cluster_points_clustering.to(torch.float32), centroid.to(torch.float32))  # Shape [n_cluster_points, 1]
            closest_idx = distances.argmin().item()

            # Get the index of the closest point in the original data
            closest_point_idx = cluster_indices[closest_idx]

            # Set the features of the closest point in X_merge to the mean of the cluster in X_merge
            merged[closest_point_idx] = cluster_points_merge.mean(dim=0)

            # Set the mask to 1 for the closest point (cluster center)
            mask[closest_point_idx] = 1

            # Set the weight of the closest point to the number of points in the cluster
            weights[closest_point_idx] = cluster_points_clustering.shape[0]

    return merged, mask, weights, None



class DPC:
    def __init__(self, dc=20, percentage=0.15):
        """
        Args:
            dc (float): Cutoff distance for local density.
            percentage (float): Fixed percentage of points to select as cluster centers.
        """
        self.dc = dc
        self.percentage = percentage  # Fixed percentage of points to select as cluster centers

    def get_clusters(self):
        if not hasattr(self, 'labels_') or self.labels_ is None:
            raise ValueError("Labels have not been initialized. Run the clustering algorithm first.")
        
        clusters = {label: np.where(self.labels_ == label)[0] for label in np.unique(self.labels_)}
        return clusters

    def fit(self, X, print_=False):
        """
        Perform Density Peaks Clustering (DPC) using a fixed percentage for cluster center selection.
        
        Args:
            X (torch.Tensor): Input data of shape [n_samples, n_features].
            print_ (bool): If True, print timing information for different steps.
        """
        n_samples = X.shape[0]
        start = time.time()

        # Use the custom distance calculation function
        dist = normal_compute_pairwise_distances(X)
        dist = torch.clamp(dist, min=0)

        if print_:
            print(f"Distance matrix computation takes {time.time() - start:.4f} seconds")

        # Zero out diagonal (distance to self is 0)
        dist.fill_diagonal_(0)

        # Compute local density (rho) using the distance cutoff (dc)
        rho = torch.exp(-(dist / self.dc) ** 2).sum(dim=1)

        if print_:
            print(f"Rho calculation takes {time.time() - start:.4f} seconds")

        # Find delta (minimum distance to a point with higher density)
        start = time.time()
        rho_expand = rho.unsqueeze(1).expand(rho.shape[0], rho.shape[0])
        rho_compare = ~torch.gt(rho_expand, rho_expand.t())
        rho_compare.fill_diagonal_(False)

        inf_mask = torch.full_like(dist, float('inf'))
        conditioned_dist = torch.where(rho_compare, dist, inf_mask)
        delta, nearest_higher_density_indices = torch.min(conditioned_dist, dim=1)

        if print_:
            print(f"Delta calculation takes {time.time() - start:.4f} seconds")

        # Compute gamma as product of delta and rho
        gamma = delta * rho
        gamma_sorted_indices = torch.argsort(-gamma)  # Sort by gamma in descending order

        # Select a fixed percentage of points as cluster centers
        num_cluster_centers = int(self.percentage * n_samples)
        cluster_centers = gamma_sorted_indices[:num_cluster_centers]

        if print_:
            print(f"Selected {len(cluster_centers)} cluster centers")
        
        use_dist_to_center = os.getenv('use_dist_to_center', "true").lower() == "true"
        
        if use_dist_to_center:
            # Original assignment: each point to the nearest cluster center
            dist_to_centers = dist[:, cluster_centers]
            nearest_center = torch.argmin(dist_to_centers, dim=1)
            labels = cluster_centers[nearest_center]
        else:
            
            # Initialize labels and assign cluster centers with their own labels
            labels = torch.full((n_samples,), -1, dtype=torch.long, device=X.device)
            labels[cluster_centers] = cluster_centers  # Assign each cluster center to itself

            # Sort points by descending order of density
            sorted_indices = torch.argsort(-rho)

            # Use precomputed nearest higher-density points
            for idx in sorted_indices:
                if labels[idx] == -1:  # If point is not yet assigned
                    nearest_higher_density_idx = nearest_higher_density_indices[idx]
                    labels[idx] = labels[nearest_higher_density_idx]
                    

        # Ensure each cluster center is labeled with its own index
        labels[cluster_centers] = cluster_centers

        # Save the labels for later retrieval
        self.labels_ = labels.cpu().numpy()

        if print_:
            print(f"Point assignment to clusters takes {time.time() - start:.4f} seconds")


def merge_DPC(tensor, clusters, repeat_with_mask=False):
    """
    Merge points in the clusters and compute the mean for each cluster using the fast mean function.

    Args:
        tensor (torch.Tensor): The input tensor to be merged.
        clusters (dict): Dictionary containing cluster indices.
        repeat_with_mask (bool): If True, use a mask for repeated clustering.

    Returns:
        output_tensor (torch.Tensor): Merged tensor with means of the clusters.
        mask (torch.Tensor): Binary mask indicating which points are cluster centers.
        weights (torch.Tensor): Weights (number of points) for each cluster.
    """
    start = time.time()

    # Use the provided fast mean calculation function
    cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(tensor, clusters)

    # Initialize output tensor, mask, and weights
    output_tensor = torch.zeros_like(tensor, dtype=cluster_means.dtype, device=tensor.device)
    mask = torch.zeros(tensor.size(0), dtype=torch.bool, device=tensor.device)
    weights = torch.zeros(tensor.size(0), dtype=cluster_means.dtype, device=tensor.device)

    # Assign means and weights to the corresponding clusters
    cluster_indices = list(clusters.keys())
    cluster_id_tensor = torch.tensor(cluster_indices, dtype=torch.long, device=tensor.device)

    weights_list = [len(clusters[cluster_id]) for cluster_id in cluster_indices]
    weights[cluster_id_tensor] = torch.tensor(weights_list, dtype=cluster_means.dtype, device=tensor.device)

    output_tensor[cluster_id_tensor] = cluster_means
    mask[cluster_id_tensor] = 1

    return output_tensor, mask, weights


def token_reduction_dpc(image_feature, image_feature_for_clustering,percentage, reduction):
    """
    Performs token reduction using Density Peaks Clustering (DPC).

    Args:
        image_feature (torch.Tensor): Tensor representing the feature to be reduced.
        image_feature_for_clustering (torch.Tensor): Tensor used for clustering.
        cutoff (float): Cutoff distance for clustering.
        reduction (list): List to track the reduction statistics.
        attentions (torch.Tensor, optional): Attention values if needed.
    """
    dpc_variant = DPC(dc=2,percentage=percentage) 
    
    
    # Fit the clustering model
    dpc_variant.fit(image_feature_for_clustering)

    # Get clusters
    clusters_variant = dpc_variant.get_clusters()
    start = time.time()

    # Merge clusters
    merged, mask_image, weights = merge_DPC(image_feature, clusters_variant, repeat_with_mask=True)
    image_feature = merged

    # reduction[1] += mask_image.sum().item()

    return image_feature, mask_image, weights, None


class Agglomerative:
    def __init__(self, n_clusters=5, linkage='average'):
        """
        Args:
            n_clusters (int): The number of clusters to find.
            linkage (str): The linkage criterion to use ('ward', 'complete', 'average', or 'single').
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def get_clusters(self):
        if not hasattr(self, 'labels_') or self.labels_ is None:
            raise ValueError("Labels have not been initialized. Run the clustering algorithm first.")
        
        clusters = {label: np.where(self.labels_ == label)[0] for label in np.unique(self.labels_)}
        return clusters

    
    def _compute_cosine_distance(self, X):
        # Normalize vectors
        X_normalized = X / X.norm(dim=1, keepdim=True)
        # Compute cosine similarity matrix
        cosine_similarity = torch.mm(X_normalized, X_normalized.T)
        # Convert to cosine distance
        cosine_distance = 1 - cosine_similarity
        return cosine_distance.cpu().numpy()

    def fit(self, X, print_=False):
        """
        Perform Agglomerative Clustering using cuML for single linkage and Scikit-learn for other linkages,
        with cosine distance for all linkages.

        Identifies cluster centers as the points closest to the mean within each cluster,
        then assigns each cluster label to its corresponding center point.

        Args:
            X (torch.Tensor): Input data of shape [n_samples, n_features].
            print_ (bool): If True, print timing information for different steps.
        """
        try :
            from cuml.cluster import AgglomerativeClustering as cumlAgglomerativeClustering
            use_cumul=True
        except :
            use_cumul=False


        n_samples = X.shape[0]
        start = time.time()

        # Compute cosine distance
        cosine_distances = self._compute_cosine_distance(X)

        if self.linkage == 'ward' and use_cumul :
            # Use cuML for single linkage clustering
            X_cpu = X.cpu().numpy()
            clustering_model = cumlAgglomerativeClustering(n_clusters=self.n_clusters, linkage='single', metric='precomputed')
            labels = clustering_model.fit_predict(cosine_distances)
        else:
            from sklearn.cluster import AgglomerativeClustering as skAgglomerativeClustering
            # Use Scikit-learn for other linkage types, with cosine distance
            
            metric="precomputed"
            clustering_model = skAgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage, metric=metric)
            labels = clustering_model.fit_predict(cosine_distances)

        if print_:
            print(f"Clustering took {time.time() - start:.4f} seconds")

        # Move labels to torch tensor on the GPU
        labels = torch.tensor(labels, dtype=torch.long, device=X.device)

        # Center identification and label assignment
        unique_labels = torch.unique(labels)
        clusters = {label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}
        cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(X, clusters)
        updated_labels = labels.clone()

        for i, (cluster_id, indices) in enumerate(clusters.items()):
            cluster_points = X[indices]
            cluster_mean = cluster_means[i]

            # Calculate cosine similarity for center identification
            cluster_points_norm = cluster_points / cluster_points.norm(dim=1, keepdim=True)
            cluster_mean_norm = cluster_mean / cluster_mean.norm()
            cosine_similarities = torch.mm(cluster_points_norm, cluster_mean_norm.unsqueeze(1)).squeeze()

            # Convert cosine similarity to cosine distance
            cosine_distances = 1 - cosine_similarities
            closest_index = indices[torch.argmin(cosine_distances)]
            
            # Assign the cluster label to the closest point
            updated_labels[indices] = closest_index

        # Update labels to reflect center-based cluster labels
        self.labels_ = updated_labels.cpu().numpy()

def merge_Agglomerative(tensor, clusters):
    """
    Merge points in the clusters and compute the mean for each cluster.
    Assign the mean to the identified cluster center, and generate a mask
    and weights for the clusters.

    Args:
        tensor (torch.Tensor): The input tensor to be merged.
        clusters (dict): Dictionary where keys are cluster center indices,
                         and values are lists or tensors of point indices in each cluster.

    Returns:
        output_tensor (torch.Tensor): Merged tensor with means of the clusters.
        mask (torch.Tensor): Binary mask indicating which points are cluster centers.
        weights (torch.Tensor): Weights (number of points) for each cluster.
    """
    start = time.time()

    # Compute the means for each cluster
    cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(tensor, clusters)

    # Extract cluster center indices as a tensor for efficient assignment
    cluster_centers = torch.tensor(list(clusters.keys()), dtype=torch.long, device=tensor.device)

    # Initialize output tensor, mask, and weights
    output_tensor = torch.zeros_like(tensor, dtype=cluster_means.dtype, device=tensor.device)
    mask = torch.zeros(tensor.size(0), dtype=torch.bool, device=tensor.device)
    weights = torch.zeros(tensor.size(0), dtype=cluster_means.dtype, device=tensor.device)

    # Assign the means, weights, and mask to the cluster centers
    output_tensor[cluster_centers] = cluster_means
    mask[cluster_centers] = 1
    weights[cluster_centers] = torch.tensor(
        [len(clusters[center.item()]) for center in cluster_centers], 
        dtype=cluster_means.dtype, 
        device=tensor.device
    )
    return output_tensor, mask, weights

def token_reduction_agglomerative(image_feature, image_feature_for_clustering, percentage_to_keep,linkage, reduction):
    """
    Performs token reduction using Agglomerative Clustering.

    Args:
        image_feature (torch.Tensor): Tensor representing the feature to be reduced.
        image_feature_for_clustering (torch.Tensor): Tensor used for clustering.
        n_clusters (int): The number of clusters to create.
        reduction (list): List to track the reduction statistics.
    """
    n_clusters=int(percentage_to_keep*image_feature.shape[0])
    agglo_variant = Agglomerative(n_clusters=n_clusters,linkage=linkage)
    
    
    # Fit the clustering model
    agglo_variant.fit(image_feature_for_clustering)

    # Get clusters
    clusters_variant = agglo_variant.get_clusters()

    # Merge clusters
    merged, mask_image, weights = merge_Agglomerative(image_feature, clusters_variant,)
    image_feature = merged

    # reduction[1] += mask_image.sum().item()

    return image_feature, mask_image, weights, None

# If you have PyTorch version >=1.12, you can use torch.segment_reduce
# Otherwise, install torch_scatter via `pip install torch-scatter`
try:
    from torch_scatter import scatter_max
    use_torch_scatter = True
except ImportError:
    use_torch_scatter = False

def compute_pairwise_distances(X):
    """
    Computes the pairwise distances between all points in X using efficient GPU operations.

    Args:
        X (torch.Tensor): Input data of shape [n_samples, n_features].

    Returns:
        distances (torch.Tensor): Pairwise distance matrix of shape [n_samples, n_samples].
    """
    # Normalize X to ensure numerical stability
    X = X / X.norm(dim=1, keepdim=True)
    # Compute pairwise distances using matrix operations
    distances = 1 - torch.mm(X, X.t())
    distances = torch.clamp(distances, min=0)
    return distances

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=1, use_cuml=True, isolate_noise_as_clusters=False):
        """
        Args:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            use_cuml (bool): Whether to use cuML for GPU-accelerated DBSCAN. Defaults to True.
            isolate_noise_as_clusters (bool): Whether to treat isolated points (noise) as their own clusters. Defaults to False.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.use_cuml = use_cuml
        self.isolate_noise_as_clusters = isolate_noise_as_clusters
        self.labels_ = None
        self.cluster_centers_ = []

    def fit(self, X, print_=False):
        """
        Perform DBSCAN clustering, using cuML if specified, otherwise a custom method.
        Identifies cluster centers as the points with the most neighbors within `eps`.

        Args:
            X (torch.Tensor): Input data of shape [n_samples, n_features].
            print_ (bool): If True, print timing information for different steps.
        """
        start = time.time()

        # Try to import cuML DBSCAN
        if self.use_cuml:
            try:
                from cuml.cluster import DBSCAN as cumlDBSCAN
                X_np = X.cpu().numpy()

                dbscan = cumlDBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean')
                self.labels_ = dbscan.fit_predict(X_np).get()

                if print_:
                    print(f"DBSCAN (cuML) clustering took {time.time() - start:.4f} seconds")
                # Center identification and label update
                dist = compute_pairwise_distances(X)
                self._assign_cluster_centers(dist)
                return
            except ImportError:
                self.use_cuml = False
                if print_:
                    print("cuML is not available. Falling back to custom implementation.")

        # Custom implementation
        X = X.to('cuda')
        n_samples = X.shape[0]

        # Compute distance matrix
        dist_start = time.time()
        dist = compute_pairwise_distances(X)
        if print_:
            print(f"Distance matrix computation took {time.time() - dist_start:.4f} seconds")

        labels = torch.full((n_samples,), -1, dtype=torch.long, device=X.device)
        cluster_id = 0

        # Identify core points
        neighbors = dist <= self.eps
        num_neighbors = neighbors.sum(dim=1)
        core_samples = num_neighbors >= self.min_samples

        # Initialize visited mask
        visited = torch.zeros(n_samples, dtype=torch.bool, device=X.device)

        # Use a queue for cluster expansion
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            if not core_samples[i]:
                labels[i] = -1  # Noise
                continue

            labels[i] = cluster_id
            seeds = neighbors[i].nonzero(as_tuple=False).squeeze()
            seeds = seeds[seeds != i]  # Remove self if present

            # Expand cluster
            queue = seeds.tolist()
            while queue:
                current_point = queue.pop()
                if visited[current_point]:
                    continue
                visited[current_point] = True
                labels[current_point] = cluster_id
                if core_samples[current_point]:
                    current_neighbors = neighbors[current_point].nonzero(as_tuple=False).squeeze()
                    current_neighbors = current_neighbors[~visited[current_neighbors]]
                    queue.extend(current_neighbors.tolist())
            cluster_id += 1

        self.labels_ = labels.cpu().numpy()

        if print_:
            print(f"DBSCAN (custom) clustering took {time.time() - start:.4f} seconds")

        # Center identification and label update
        self._assign_cluster_centers(dist)

    def _assign_cluster_centers(self, dist):
        """
        Identifies cluster centers based on the most connected points in each cluster and updates the labels.
        Noise points are treated as their own clusters if isolate_noise_as_clusters is enabled.
        """
        labels = torch.tensor(self.labels_, device=dist.device)
        n_samples = labels.size(0)

        if not self.isolate_noise_as_clusters:
            # Exclude noise points
            valid_mask = labels != -1
            labels_valid = labels[valid_mask]
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze()
        else:
            labels_valid = labels
            valid_indices = torch.arange(n_samples, device=dist.device)

        unique_labels, labels_inverse = torch.unique(labels_valid, return_inverse=True)
        cluster_indices = labels_inverse  # Cluster indices from 0 to n_clusters-1

        # Get the distance submatrix for valid indices
        dist_valid = dist[valid_indices][:, valid_indices]

        # Compute adjacency within eps
        adjacency = dist_valid <= self.eps

        # Compute neighbors within eps
        neighbors_within_eps = adjacency.sum(dim=1)

        # Now, compute per-cluster maximums
        if use_torch_scatter:
            # Use torch_scatter's scatter_max
            max_neighbors, argmax_indices = scatter_max(
                neighbors_within_eps, cluster_indices, dim=0
            )
        else:
            # Use torch.segment_reduce in PyTorch >=1.12
            try:
                max_neighbors = torch.zeros(unique_labels.size(0), device=dist.device)
                argmax_indices = torch.zeros(unique_labels.size(0), dtype=torch.long, device=dist.device)

                # Sort cluster_indices and neighbors_within_eps
                sorted_indices = cluster_indices.argsort()
                sorted_clusters = cluster_indices[sorted_indices]
                sorted_neighbors = neighbors_within_eps[sorted_indices]

                # Use torch.repeat_interleave to get counts
                counts = torch.bincount(sorted_clusters)

                # Compute segment offsets
                segment_offsets = torch.cat([torch.tensor([0], device=dist.device), counts.cumsum(0)[:-1]])

                for i in range(unique_labels.size(0)):
                    start = segment_offsets[i]
                    end = segment_offsets[i] + counts[i]
                    segment = sorted_neighbors[start:end]
                    max_val, argmax = segment.max(0)
                    max_neighbors[i] = max_val
                    argmax_indices[i] = sorted_indices[start + argmax]
            except AttributeError:
                raise RuntimeError("Please install torch_scatter or use PyTorch >=1.12 to enable this optimization.")

        # Get the cluster center indices in the valid_indices array
        cluster_center_indices_in_valid = argmax_indices

        # Map back to original indices
        cluster_center_indices = valid_indices[cluster_center_indices_in_valid]

        # Update labels
        updated_labels = labels.clone()
        updated_labels[valid_indices] = cluster_center_indices[cluster_indices]

        if self.isolate_noise_as_clusters:
            # Handle noise points
            noise_mask = labels == -1
            noise_indices = noise_mask.nonzero(as_tuple=False).squeeze()
            updated_labels[noise_indices] = noise_indices

        # Update self.labels_
        self.labels_ = updated_labels.cpu().numpy()

    def get_clusters(self):
        """
        Returns a dictionary of clusters, with cluster center indices as keys and point indices as values.
        """
        if self.labels_ is None:
            raise ValueError("Labels have not been initialized. Run the clustering algorithm first.")

        clusters = {}
        for idx, center in enumerate(self.labels_):
            if center == -1:
                continue
            clusters.setdefault(center, []).append(idx)
        return clusters

def merge_DBSCAN(tensor, dbscan_variant):
    """
    Merges points in the clusters and computes the mean for each cluster using identified cluster centers.

    Args:
        tensor (torch.Tensor): The input tensor to be merged.
        dbscan_variant (DBSCAN): A DBSCAN object containing labels and cluster centers.

    Returns:
        output_tensor (torch.Tensor): Merged tensor with means of the clusters.
        mask (torch.Tensor): Binary mask indicating which points are cluster centers.
        weights (torch.Tensor): Weights (number of points) for each cluster.
    """
    clusters = dbscan_variant.get_clusters()

    start = time.time()

    # Use the provided function to compute cluster means
    cluster_means = compute_fastest_cluster_means_with_arbitrary_ids(tensor, clusters)

    # Initialize output tensor, mask, and weights
    n_samples = tensor.size(0)
    output_tensor = torch.zeros_like(tensor, dtype=cluster_means.dtype, device=tensor.device)
    mask = torch.zeros(n_samples, dtype=torch.bool, device=tensor.device)
    weights = torch.zeros(n_samples, dtype=cluster_means.dtype, device=tensor.device)

    # Assign means and weights to the corresponding clusters
    cluster_indices = list(clusters.keys())
    cluster_centers = torch.tensor(cluster_indices, dtype=torch.long, device=tensor.device)

    # Prepare weights
    weights_list = [len(clusters[cluster_id]) for cluster_id in cluster_indices]
    weights[cluster_centers] = torch.tensor(weights_list, dtype=cluster_means.dtype, device=tensor.device)

    # Assign cluster means to output tensor at cluster center positions
    output_tensor[cluster_centers] = cluster_means
    mask[cluster_centers] = True

    return output_tensor, mask, weights


def token_reduction_dbscan(image_feature, image_feature_for_clustering, eps, reduction, min_samples=5, use_cuml=True, isolate_noise_as_clusters=False):
    """
    Performs token reduction using DBSCAN.

    Args:
        image_feature (torch.Tensor): Tensor representing the feature to be reduced.
        image_feature_for_clustering (torch.Tensor): Tensor used for clustering.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        reduction (list): List to track the reduction statistics.
        use_cuml (bool): Whether to use cuML for GPU-accelerated DBSCAN. Defaults to True.
        isolate_noise_as_clusters (bool): Whether to treat isolated points as their own clusters. Defaults to False.
    """
    # Initialize the DBSCAN instance
    dbscan_variant = DBSCAN(eps=eps, min_samples=min_samples, use_cuml=use_cuml, isolate_noise_as_clusters=isolate_noise_as_clusters)

    # Fit the clustering model
    dbscan_variant.fit(image_feature_for_clustering)
    

    # Merge clusters with the point having most neighbors as cluster center
    merged, mask_image, weights = merge_DBSCAN(image_feature, dbscan_variant)
    image_feature = merged

    # reduction[1] += mask_image.sum().item()

    return image_feature, mask_image, weights, None
