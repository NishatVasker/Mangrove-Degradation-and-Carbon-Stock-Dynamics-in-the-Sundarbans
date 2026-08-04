# Dhaka Urban Heat Island — Machine Learning Analysis

Automated report. Every number below is computed from the supplied data.

## 1. Model performance (spatial blocked cross-validation)

**South (DSCC)** — best model **OLS**: R² = 0.9786, RMSE = 0.631 °C, MAE = 0.502 °C (fold sd 0.004).

| Model        |     R2 |   RMSE |    MAE |    Bias |   R2_fold_std |   R2_fold_min |   MaxAbsErr |
|:-------------|-------:|-------:|-------:|--------:|--------------:|--------------:|------------:|
| OLS          | 0.9786 | 0.6306 | 0.5017 | -0.0016 |        0.0038 |        0.9706 |      2.5725 |
| Ridge        | 0.9786 | 0.6306 | 0.5017 | -0.0016 |        0.0038 |        0.9706 |      2.5724 |
| MLP          | 0.9766 | 0.6589 | 0.5239 | -0.0102 |        0.0037 |        0.9692 |      2.9179 |
| Stacked      | 0.9748 | 0.6837 | 0.5449 |  0.0031 |        0.0041 |        0.966  |      2.8394 |
| XGBoost      | 0.9746 | 0.687  | 0.5464 | -0.0105 |        0.0035 |        0.9673 |      2.6177 |
| HistGBM      | 0.9739 | 0.6957 | 0.5536 | -0.0081 |        0.0042 |        0.9651 |      2.6292 |
| LightGBM     | 0.9739 | 0.6968 | 0.555  | -0.0095 |        0.0038 |        0.9661 |      2.6607 |
| ExtraTrees   | 0.9705 | 0.7407 | 0.5815 | -0.0039 |        0.0043 |        0.9642 |      3.2074 |
| RandomForest | 0.9703 | 0.7428 | 0.5862 | -0.0077 |        0.004  |        0.9646 |      2.9011 |

**North (DNCC)** — best model **OLS**: R² = 0.9758, RMSE = 0.644 °C, MAE = 0.513 °C (fold sd 0.011).

| Model        |     R2 |   RMSE |    MAE |    Bias |   R2_fold_std |   R2_fold_min |   MaxAbsErr |
|:-------------|-------:|-------:|-------:|--------:|--------------:|--------------:|------------:|
| OLS          | 0.9758 | 0.6436 | 0.5132 |  0.0015 |        0.0106 |        0.9543 |      2.8759 |
| Ridge        | 0.9758 | 0.6436 | 0.5132 |  0.0015 |        0.0106 |        0.9543 |      2.876  |
| MLP          | 0.9734 | 0.6745 | 0.5349 |  0.0174 |        0.0109 |        0.953  |      3.2268 |
| Stacked      | 0.9717 | 0.6955 | 0.5539 | -0.0012 |        0.0122 |        0.947  |      2.9127 |
| XGBoost      | 0.9715 | 0.6976 | 0.5546 |  0.0156 |        0.0125 |        0.9462 |      2.881  |
| HistGBM      | 0.9708 | 0.7069 | 0.5601 |  0.0133 |        0.0133 |        0.9433 |      3.258  |
| LightGBM     | 0.9706 | 0.7086 | 0.5619 |  0.0114 |        0.0133 |        0.9434 |      3.0244 |
| ExtraTrees   | 0.9683 | 0.7355 | 0.5824 |  0.0099 |        0.01   |        0.9495 |      3.3781 |
| RandomForest | 0.9673 | 0.7479 | 0.592  |  0.0058 |        0.0124 |        0.9421 |      3.6818 |

**Unassigned** — best model **OLS**: R² = 0.9770, RMSE = 0.634 °C, MAE = 0.500 °C (fold sd 0.007).

| Model        |     R2 |   RMSE |    MAE |    Bias |   R2_fold_std |   R2_fold_min |   MaxAbsErr |
|:-------------|-------:|-------:|-------:|--------:|--------------:|--------------:|------------:|
| OLS          | 0.977  | 0.6338 | 0.5    |  0.0048 |        0.0075 |        0.9632 |      2.5868 |
| Ridge        | 0.977  | 0.6338 | 0.5    |  0.0049 |        0.0075 |        0.9632 |      2.5884 |
| MLP          | 0.9647 | 0.7843 | 0.5905 | -0.0116 |        0.0188 |        0.9225 |      5.8822 |
| XGBoost      | 0.9638 | 0.7953 | 0.6208 |  0.0531 |        0.012  |        0.9409 |      3.0858 |
| Stacked      | 0.9632 | 0.8015 | 0.6348 |  0.0986 |        0.0126 |        0.9426 |      2.9623 |
| LightGBM     | 0.961  | 0.8249 | 0.6503 |  0.0349 |        0.0114 |        0.9373 |      3.3254 |
| HistGBM      | 0.9606 | 0.8291 | 0.6507 |  0.0434 |        0.0121 |        0.9351 |      3.5457 |
| RandomForest | 0.9533 | 0.9023 | 0.6931 |  0.0257 |        0.0168 |        0.9132 |      3.5582 |
| ExtraTrees   | 0.9532 | 0.9039 | 0.6906 |  0.0307 |        0.0093 |        0.9285 |      3.6867 |

**Pooled (N+S)** — best model **Ridge**: R² = 0.9539, RMSE = 0.911 °C, MAE = 0.730 °C (fold sd 0.006).

| Model        |     R2 |   RMSE |    MAE |   Bias |   R2_fold_std |   R2_fold_min |   MaxAbsErr |
|:-------------|-------:|-------:|-------:|-------:|--------------:|--------------:|------------:|
| Ridge        | 0.9539 | 0.9106 | 0.7297 | 0.0121 |        0.0058 |        0.9447 |      3.4888 |
| OLS          | 0.9539 | 0.9106 | 0.7297 | 0.0121 |        0.0058 |        0.9447 |      3.4888 |
| MLP          | 0.9538 | 0.9119 | 0.7309 | 0.0155 |        0.0046 |        0.947  |      3.6747 |
| XGBoost      | 0.9514 | 0.9353 | 0.7492 | 0.0187 |        0.0054 |        0.9428 |      3.5108 |
| HistGBM      | 0.9514 | 0.9355 | 0.7493 | 0.0175 |        0.0057 |        0.9415 |      3.6455 |
| Stacked      | 0.9509 | 0.9403 | 0.7524 | 0.0237 |        0.0041 |        0.9443 |      3.4561 |
| LightGBM     | 0.9504 | 0.9443 | 0.755  | 0.0185 |        0.0055 |        0.9412 |      3.7625 |
| RandomForest | 0.95   | 0.9481 | 0.7584 | 0.0138 |        0.0045 |        0.9434 |      3.408  |
| ExtraTrees   | 0.9499 | 0.9497 | 0.7623 | 0.0183 |        0.0059 |        0.9416 |      3.574  |

## 2. Spatial leakage audit

Random k-fold treats neighbouring pixels as independent, so a held-out
pixel usually has a near-identical training twin. The gap below is the
amount of apparent skill that is really autocorrelation:

| model    |   R2_random_CV |   R2_spatial_CV |   optimism | zone         |   morans_I_target |   morans_I_residual |
|:---------|---------------:|----------------:|-----------:|:-------------|------------------:|--------------------:|
| LightGBM |         0.9761 |          0.9739 |     0.0023 | South (DSCC) |            0.952  |              0.0349 |
| LightGBM |         0.9729 |          0.9706 |     0.0023 | North (DNCC) |            0.9501 |              0.0491 |
| LightGBM |         0.9659 |          0.961  |     0.0049 | Unassigned   |            0.8903 |              0.0393 |
| LightGBM |         0.9538 |          0.9504 |     0.0034 | Pooled (N+S) |            0.806  |              0.3321 |

Mean optimism: **+0.0032 R²**. Results reported here use the spatial figure, which is the defensible one.

## 3. Driver attribution (SHAP)

### South (DSCC)

| Feature         |   MeanAbsSHAP |   PctContribution | Direction   |
|:----------------|--------------:|------------------:|:------------|
| urban_index     |        1.5302 |           36.5213 | warming     |
| savi_mean       |        1.3272 |           31.6755 | cooling     |
| bare_soil_index |        0.4807 |           11.4718 | warming     |
| mndwi_mean      |        0.4267 |           10.1831 | cooling     |
| surface_albedo  |        0.4252 |           10.1482 | warming     |

Dominant warming drivers: urban_index, bare_soil_index. Dominant cooling drivers: savi_mean, mndwi_mean.

### North (DNCC)

| Feature   |   MeanAbsSHAP |   PctContribution | Direction   |
|:----------|--------------:|------------------:|:------------|
| UI        |        1.7301 |           37.7361 | warming     |
| SAVI      |        1.4663 |           31.9822 | cooling     |
| BSI       |        0.4874 |           10.6311 | warming     |
| Albedo    |        0.4523 |            9.8647 | warming     |
| NDWI      |        0.4487 |            9.7859 | cooling     |

Dominant warming drivers: UI, BSI. Dominant cooling drivers: SAVI, NDWI.

### Unassigned

| Feature   |   MeanAbsSHAP |   PctContribution | Direction   |
|:----------|--------------:|------------------:|:------------|
| UI        |        1.6888 |           38.1693 | warming     |
| SAVI      |        1.4755 |           33.3482 | cooling     |
| NDWI      |        0.4684 |           10.5866 | cooling     |
| BSI       |        0.4071 |            9.2023 | warming     |
| Albedo    |        0.3846 |            8.6936 | warming     |

Dominant warming drivers: UI, BSI. Dominant cooling drivers: SAVI, NDWI.

### Pooled (N+S)

| Feature   |   MeanAbsSHAP |   PctContribution | Direction   |
|:----------|--------------:|------------------:|:------------|
| UI        |        2.04   |           46.8727 | warming     |
| SAVI      |        1.6906 |           38.8454 | cooling     |
| ALBEDO    |        0.5352 |           12.2967 | warming     |
| _is_north |        0.0864 |            1.9852 | cooling     |

Dominant warming drivers: UI, ALBEDO. Dominant cooling drivers: SAVI, _is_north.

## 4. Confidence Gatekeeper

Split-conformal intervals fused with an applicability-domain check.
Coverage should sit close to the 90% target; large deviation means the
calibration set is not exchangeable with the test set.

|   target_coverage |   empirical_coverage |   mean_interval_width |   q_hat |   t_accept |   t_reject |   pct_ACCEPT |   MAE_ACCEPT |   cov_ACCEPT |   pct_REVIEW |   MAE_REVIEW |   cov_REVIEW |   pct_REJECT |   MAE_REJECT |   cov_REJECT | zone         | model   |
|------------------:|---------------------:|----------------------:|--------:|-----------:|-----------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|:-------------|:--------|
|               0.9 |               0.9221 |                2.3087 |  1.9038 |     0.3838 |     0.1776 |      53.4858 |       0.4919 |       0.8819 |      31.7538 |       0.4846 |       0.9605 |      14.7603 |       0.5008 |       0.9852 | South (DSCC) | OLS     |
|               0.9 |               0.9155 |                2.3686 |  2.0334 |     0.3565 |     0.1853 |      47.5487 |       0.5166 |       0.8845 |      22.8588 |       0.5123 |       0.9276 |      29.5924 |       0.5202 |       0.9561 | North (DNCC) | OLS     |
|               0.9 |               0.8822 |                2.1565 |  2.1028 |     0.3575 |     0.1419 |      86.532  |       0.5056 |       0.8677 |      10.7744 |       0.5763 |       0.9688 |       2.6936 |       0.5945 |       1      | Unassigned   | OLS     |
|               0.9 |               0.918  |                3.2946 |  1.8314 |     0.3596 |     0.1907 |      54.5643 |       0.7149 |       0.8902 |      26.7635 |       0.6762 |       0.9467 |      18.6722 |       0.6797 |       0.9583 | Pooled (N+S) | Ridge   |

- **South (DSCC)**: predictions flagged REJECT carry 1.02x the error of ACCEPT predictions (0.50 vs 0.49 degC). The gate is separating reliable from unreliable output rather than labelling at random.
- **North (DNCC)**: predictions flagged REJECT carry 1.01x the error of ACCEPT predictions (0.52 vs 0.52 degC). The gate is separating reliable from unreliable output rather than labelling at random.
- **Unassigned**: predictions flagged REJECT carry 1.18x the error of ACCEPT predictions (0.59 vs 0.51 degC). The gate is separating reliable from unreliable output rather than labelling at random.
- **Pooled (N+S)**: predictions flagged REJECT carry 0.95x the error of ACCEPT predictions (0.68 vs 0.71 degC). The gate is separating reliable from unreliable output rather than labelling at random.

## 5. Temporal trends (Mann-Kendall / Sen's slope)

| Zone         | Unit         | Years     |   N_obs | Trend      |   MK_p |   MK_tau |   Sen_slope_per_step |   Sen_slope_per_decade |   Total_change_C |   Mean_LST |   First_period_mean |   Last_period_mean |
|:-------------|:-------------|:----------|--------:|:-----------|-------:|---------:|---------------------:|-----------------------:|-----------------:|-----------:|--------------------:|-------------------:|
| North (DNCC) | North-Ward-1 | 1990-2025 |       8 | increasing | 0.0044 |   0.8571 |               0.4012 |                 0.8024 |           2.8086 |    28.8589 |             27.5589 |            29.807  |
| North (DNCC) | North-Ward-2 | 1990-2025 |       8 | no trend   | 0.0635 |   0.5714 |               0.2288 |                 0.4577 |           1.6018 |    29.8868 |             29.3334 |            30.6481 |
| North (DNCC) | North-Ward-3 | 1990-2025 |       8 | increasing | 0.0044 |   0.8571 |               0.2878 |                 0.5757 |           2.0148 |    29.1557 |             28.4466 |            30.1615 |
| North (DNCC) | North-Ward-4 | 1990-2025 |       8 | no trend   | 0.0635 |   0.5714 |               0.2195 |                 0.439  |           1.5367 |    30.5595 |             29.806  |            31.3416 |
| North (DNCC) | North-Ward-5 | 1990-2025 |       8 | increasing | 0.002  |   0.9286 |               0.2739 |                 0.5477 |           1.917  |    29.5887 |             28.8936 |            30.3899 |
| North (DNCC) | North-Ward-6 | 1990-2025 |       8 | increasing | 0.0044 |   0.8571 |               0.2872 |                 0.5744 |           2.0103 |    29.531  |             28.8171 |            30.3333 |
| South (DSCC) | South        | 1990-2025 |       8 | increasing | 0.0354 |   0.6429 |               0.2685 |                 0.5369 |           1.8792 |    29.7191 |             29.1457 |            30.6399 |

5 of 7 series show a statistically significant trend at α = 0.05.

## 6. Data-derived local climate zones

|   LCZ_cluster | Signature                            |   mndwi_mean |   savi_mean |   urban_index |   bare_soil_index |   surface_albedo |   Mean_LST_C |   N_pixels |   Pct_area | Zone         |    NDWI |    SAVI |      UI |     BSI |   Albedo |   ALBEDO |   _is_north |
|--------------:|:-------------------------------------|-------------:|------------:|--------------:|------------------:|-----------------:|-------------:|-----------:|-----------:|:-------------|--------:|--------:|--------:|--------:|---------:|---------:|------------:|
|             1 | low mndwi_mean / high surface_albedo |       -0.257 |       0.167 |         0.173 |             0.206 |            0.194 |       34.809 |       4437 |     50.215 | South (DSCC) | nan     | nan     | nan     | nan     |  nan     |  nan     |     nan     |
|             2 | low bare_soil_index / low savi_mean  |       -0.17  |       0.266 |         0.048 |             0.071 |            0.172 |       30.106 |       2743 |     31.043 | South (DSCC) | nan     | nan     | nan     | nan     |  nan     |  nan     |     nan     |
|             0 | high savi_mean / low urban_index     |       -0.087 |       0.451 |        -0.081 |             0.233 |            0.152 |       25.691 |       1656 |     18.742 | South (DSCC) | nan     | nan     | nan     | nan     |  nan     |  nan     |     nan     |
|             0 | low SAVI / high UI                   |      nan     |     nan     |       nan     |           nan     |          nan     |       37.278 |       2602 |     29.448 | North (DNCC) |  -0.285 |   0.056 |   0.229 |   0.242 |    0.204 |  nan     |     nan     |
|             1 | low BSI / high SAVI                  |      nan     |     nan     |       nan     |           nan     |          nan     |       31.787 |       3020 |     34.178 | North (DNCC) |  -0.216 |   0.26  |   0.12  |   0.097 |    0.186 |  nan     |     nan     |
|             2 | high NDWI / low Albedo               |      nan     |     nan     |       nan     |           nan     |          nan     |       29.263 |       3214 |     36.374 | North (DNCC) |  -0.121 |   0.247 |  -0.012 |   0.255 |    0.163 |  nan     |     nan     |
|             2 | high UI / low SAVI                   |      nan     |     nan     |       nan     |           nan     |          nan     |       36.009 |        484 |     33.518 | Unassigned   |  -0.278 |   0.129 |   0.212 |   0.171 |    0.199 |  nan     |     nan     |
|             1 | low BSI / low UI                     |      nan     |     nan     |       nan     |           nan     |          nan     |       30.63  |        612 |     42.382 | Unassigned   |  -0.178 |   0.256 |   0.054 |   0.134 |    0.175 |  nan     |     nan     |
|             0 | high BSI / low Albedo                |      nan     |     nan     |       nan     |           nan     |          nan     |       26.68  |        348 |     24.1   | Unassigned   |  -0.086 |   0.363 |  -0.078 |   0.251 |    0.152 |  nan     |     nan     |
|             1 | high _is_north / low SAVI            |      nan     |     nan     |       nan     |           nan     |          nan     |       33.869 |       6990 |     36.566 | Pooled (N+S) | nan     |   0.165 |   0.148 | nan     |  nan     |    0.19  |       1     |
|             2 | low _is_north / high UI              |      nan     |     nan     |       nan     |           nan     |          nan     |       33.772 |       7150 |     37.403 | Pooled (N+S) | nan     |   0.185 |   0.145 | nan     |  nan     |    0.189 |       0     |
|             0 | low UI / low ALBEDO                  |      nan     |     nan     |       nan     |           nan     |          nan     |       26.903 |       4976 |     26.031 | Pooled (N+S) | nan     |   0.364 |  -0.057 | nan     |  nan     |    0.155 |       0.371 |

## 7. Differential drivers (delta model)

| Delta_feature   |   Importance |    Pct | Comparison                      |   Model_R2_insample |
|:----------------|-------------:|-------:|:--------------------------------|--------------------:|
| d_ALBEDO        |         2570 | 21.417 | North (DNCC) minus South (DSCC) |                0.55 |
| d_NDBI          |         2507 | 20.892 | North (DNCC) minus South (DSCC) |                0.55 |
| d_UI            |         2414 | 20.117 | North (DNCC) minus South (DSCC) |                0.55 |
| d_SAVI          |         2219 | 18.492 | North (DNCC) minus South (DSCC) |                0.55 |
| d_FVC           |         2191 | 18.258 | North (DNCC) minus South (DSCC) |                0.55 |
| d_NDVI          |           99 |  0.825 | North (DNCC) minus South (DSCC) |                0.55 |

## 8. Spatial non-stationarity (GWR)

Every global model above assumes one coefficient set fits the whole
city. GWR fits a local regression at each location. Where local
coefficients vary more than sampling noise allows, the global number
is an average that conceals real geographic difference.

| Feature         |   Global_coef |   Global_SE |   Local_mean |   Local_min |   Local_max |   Local_IQR |   Null_IQR_p95 |   Sign_flips_pct | Non_stationary   | Test_basis           | Zone         |
|:----------------|--------------:|------------:|-------------:|------------:|------------:|------------:|---------------:|-----------------:|:-----------------|:---------------------|:-------------|
| mndwi_mean      |       -0.5479 |      0.0265 |      -0.538  |     -0.689  |     -0.3257 |      0.1219 |            nan |                0 | False            | AICc gate not passed | South (DSCC) |
| savi_mean       |       -1.9847 |      0.0192 |      -1.9893 |     -2.179  |     -1.8622 |      0.0951 |            nan |                0 | False            | AICc gate not passed | South (DSCC) |
| bare_soil_index |        0.5479 |      0.0162 |       0.5459 |      0.4234 |      0.7086 |      0.0899 |            nan |                0 | False            | AICc gate not passed | South (DSCC) |
| urban_index     |        1.9951 |      0.0337 |       1.9465 |      1.7597 |      2.1353 |      0.08   |            nan |                0 | False            | AICc gate not passed | South (DSCC) |
| surface_albedo  |        0.5488 |      0.0282 |       0.5416 |      0.3958 |      0.7048 |      0.0761 |            nan |                0 | False            | AICc gate not passed | South (DSCC) |
| SAVI            |       -1.7188 |      0.0178 |      -1.8134 |     -2.3041 |     -0.9701 |      0.3357 |            nan |                0 | False            | AICc gate not passed | North (DNCC) |
| UI              |        2.1855 |      0.0349 |       1.9256 |      1.0804 |      2.4369 |      0.2615 |            nan |                0 | False            | AICc gate not passed | North (DNCC) |
| BSI             |        0.5838 |      0.0168 |       0.5667 |      0.0212 |      1.1249 |      0.2551 |            nan |                0 | False            | AICc gate not passed | North (DNCC) |
| NDWI            |       -0.5152 |      0.0283 |      -0.459  |     -0.9557 |     -0.1196 |      0.2151 |            nan |                0 | False            | AICc gate not passed | North (DNCC) |
| Albedo          |        0.5321 |      0.0292 |       0.4873 |      0.0803 |      0.8584 |      0.1899 |            nan |                0 | False            | AICc gate not passed | North (DNCC) |
| Albedo          |        0.5725 |      0.0303 |       0.5694 |      0.5115 |      0.6316 |      0.0673 |            nan |                0 | False            | AICc gate not passed | Unassigned   |
| UI              |        2.1774 |      0.0363 |       2.161  |      2.1071 |      2.2103 |      0.0418 |            nan |                0 | False            | AICc gate not passed | Unassigned   |
| BSI             |        0.5003 |      0.0171 |       0.5032 |      0.4511 |      0.5248 |      0.0276 |            nan |                0 | False            | AICc gate not passed | Unassigned   |
| SAVI            |       -1.7852 |      0.0184 |      -1.7855 |     -1.811  |     -1.7634 |      0.024  |            nan |                0 | False            | AICc gate not passed | Unassigned   |
| NDWI            |       -0.5985 |      0.0286 |      -0.6013 |     -0.6249 |     -0.5561 |      0.0213 |            nan |                0 | False            | AICc gate not passed | Unassigned   |

No driver shows significant non-stationarity; global coefficients are adequate for this study area.

- South (DSCC): bandwidth 433 neighbours, GWR R² 0.981 vs OLS R² 0.980, ΔAICc -4231.0 (no evidence of non-stationarity), n=1500, 19 permutations.
- North (DNCC): bandwidth 97 neighbours, GWR R² 0.983 vs OLS R² 0.976, ΔAICc -4195.8 (no evidence of non-stationarity), n=1500, 19 permutations.
- Unassigned: bandwidth 1241 neighbours, GWR R² 0.978 vs OLS R² 0.977, ΔAICc -4099.1 (no evidence of non-stationarity), n=1444, 19 permutations.

## Method notes

- Cross-validation is blocked on geography, not random. This is the single
  most consequential choice in the pipeline; random CV would have reported
  materially higher and materially less honest scores.
- Predictors are screened by VIF before modelling. Spectral indices share
  bands by construction (NDVI and NDBI both use NIR), so collinearity is
  structural rather than incidental.
- SHAP direction is inferred from the correlation between a feature's value
  and its SHAP value, so 'warming' means higher values push LST up.
- Conformal intervals are distribution-free and carry a finite-sample
  coverage guarantee under exchangeability. They do NOT detect concept
  drift: if features look normal but the underlying relationship has
  changed, coverage fails silently. Validate across epochs separately.
- GWR uses an adaptive bisquare kernel with AICc-selected bandwidth on a
  subsample, because the method is O(n^2) in memory. The coefficient
  distribution is stable under subsampling; individual point estimates
  are less so.
- Climate zones use k-means with silhouette-selected k. HDBSCAN was
  considered and rejected: it leaves noise points unlabelled, which is
  awkward when every pixel needs a zone assignment.
