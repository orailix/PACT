# Configuration Options

## General Settings
- **visual_token_reduction** *(default: False)*: Enables visual token reduction. If False, all other variables will not be taken into account and regular inference is applied.
- **layer_for_reduction** *(default: 4)*: Layer index where token reduction is applied.
- **progessive_reduction** *(default: False)*: If True, progressively reduces tokens across layers.
- **log_output_path** *(default: "aggregated_metrics")*: Directory to save performance logs.
- **synchro** *(default: False)*: Synchronization for accurate computation of inference speed metrics.

## Clustering-based merging general variables
- **vector_to_use_in_distance_clustering** *(default: "current_k_cosine")*: Metric to use when computing distances between tokens (e.g., "current_k_cosine" for using Key vectors after the application of rotary embedding; other alternatives are current_k, current_q, and current_q_cosine).
- **include_pruned_in_mean** *(default: True)*: For PACT, this variable means the retrieval step is active. If a custom merging function is used, pruned vectors are also fed to the reduction function as input.

## PACT
*Note: Clustering variables above and pruning variables below affect PACT.*
- **coef_pruned** *(default: 1.5)*: The alpha variable for PACT.
- **include_pruned_in_mean** *(default: True)*: If True, the retrieval step is active. Affects only PACT.
- **take_mean** *(default: True)*: Whether to compute the mean feature for clustered tokens. Affects DBDPC (so PACT) and ToMe (other clustering-based approaches take the mean by default).
- **get_mean_position_id** *(default: False)*: Uses the average position ID for DBDPC.

## TOME
- **use_tome** *(default: False)*: Enables ToMe.
- **perc_tokeep_tome_total** *(default: 1.0)* and **tome_equivalant_layer_for_reduction** *(default: 4)*: Used to determine ToMe's scheduler so that the average number of visual tokens across layers matches a single-layer reduction approach (like PACT) using the same parameters.

## KMeans Reduction
- **use_kmeans** *(default: False)*: Enables token clustering with k-means.
- **perc_tokeep_kmeans** *(default: 1.0)*: Percentage of tokens to keep.

## DPC (Density-Peak Clustering)
- **use_dpc** *(default: False)*: Enables DPC.
- **percentage_to_keep_dpc** *(default: 1.0)*: Percentage of tokens to keep with DPC.

## Agglomerative Clustering
- **use_agglomerative** *(default: False)*: Enables agglomerative clustering.
- **percentage_to_keep_agglomerative** *(default: 1.0)*: Percentage of tokens to keep.
- **linkage** *(default: "single")*: Linkage method used (e.g., "single", "complete").

## DBSCAN
- **use_dbscan** *(default: False)*: Enables DBSCAN clustering.
- **eps_dbscan** *(default: 0.1)*: Epsilon value for DBSCAN clustering.
- **noise_as_clusters_dbscan** *(default: False)*: Treat noise points as separate clusters.

## Token Pruning
- **token_pruning** *(default: False)*: Enables token pruning (removal based on importance metric).
- **prune_with_norm** *(default: False)*: Use only the norm as an importance metric.
- **use_cosine_in_token_pruning** *(default: False)*: Apply rotary embeddings to keys and queries before using them to determine the importance metric.
- **use_attention_in_token_pruning** *(default: False)*: Use attention-based importance (usage of attention mask or rotary embedding is determined by other variables).
- **use_mask_in_use_attention_in_token_pruning** *(default: False)*: When use_attention_in_token_pruning is True, uses attention mask when computing attention scores.
- **pruning_filter_wth_percentage** *(default: True)*: Whether to keep tokens based on a percentage.
- **pruning_tokeep_percentage_value** *(default: 1.0)*: Percentage of tokens to retain.
- **use_IQR_in_token_pruning** *(default: False)*: Uses Interquartile Range to filter outliers.
- **alpha_IQR** *(default: 0.5)*: Alpha value for IQR-based pruning.
- **do_not_upcast_to_full_precision_for_pruning** *(default: False)*: Do not upcast tensors to compute the importance metric.

## VTW (Visual Token Withdrawal)
- **withdraw_visual_tokens** *(default: False)*: Withdraws visual tokens.
- **VTW_equivalant_layer_for_reduction** *(default: -1)* and **equivalent_reduc_percentage_vtw** *(default: 0.0)*: Used to determine the layer at which visual tokens will be withdrawn so that the average number of visual tokens across layers matches a single-layer reduction approach (like PACT) using the same parameters.

## Positional IDs Handling
- **change_position_ids** *(default: False)*: Adjusts token position IDs to fill pruned tokens. For example, if the resulting positional IDs are 0,7,8,10, activating this variable will change them to 0,1,2,3.

## Attention Calculation
- **keep_casual** *(default: True)*: Whether to keep the causal variable as True or add the triangular mask to the proportional attention weights. Results are very similar if this is activated or not.
- **no_proportional_attention** *(default: False)*: Disables proportional attention.

## Metrics & Logging
- **get_performance_metrics** *(default: False)*: Whether to compute accuracy/performance.
- **get_reduction_ratio** *(default: False)*: Whether to log the reduction ratio. Automatically activated if get_performance_metrics is True.

