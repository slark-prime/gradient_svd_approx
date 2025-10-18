## svd_optimality
- fro_svd: 3.711286
- fro_randomized: 3.711286
- fro_cur: 12.732485
- spectral_svd: 2.502318
- spectral_randomized: 2.502318
- spectral_cur: 10.798455

## weighted_geometry
- weighted_error_vanilla: 31.725132
- weighted_error_whitened: 28.301728
- fro_vanilla: 1.742623
- fro_whitened: 1.978550

## descent_alignment
- weighted_cosine_vanilla: 0.995320
- weighted_cosine_whitened: 0.997520
- weighted_drop_vanilla: 40084.165898
- weighted_drop_whitened: 40223.626508
- standard_cosine_vanilla: 0.981269
- standard_cosine_whitened: 0.975287

## robustness
- l1_to_clean_vanilla: 423.972395
- l1_to_clean_trimmed: 30.993471
- fro_to_clean_vanilla: 265.083467
- fro_to_clean_trimmed: 23.985910
- cosine_to_clean_vanilla: 0.014132
- cosine_to_clean_trimmed: 0.095105

## tensor
- relative_error_to_clean_matricized: 72.770164
- relative_error_to_clean_hosvd: 47.212250
- relative_error_to_observed_matricized: 0.000398
- relative_error_to_observed_hosvd: 0.761176
- avg_angle_deg_matricized: 58.712983
- avg_angle_deg_hosvd: 58.712978

## streaming
- cumulative_error_fd: 86.498146
- cumulative_error_static: 718.485744
- cumulative_error_periodic_svd: 858.368821
- cumulative_error_optimal_per_step: 36.709529
- fd_shrink_calls: 13
- periodic_svd_recomputes: 8
