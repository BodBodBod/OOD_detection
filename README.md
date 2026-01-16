# OOD Detection in Tabular Data

This repository is dedicated to research in the field of Out-of-Distribution (OOD) detection on tabular data. It explores and implements two methodological approaches for identifying distributional shifts: a **subsample-wise** approach for detecting OOD samples and a **point-wise** approach for detecting individual OOD data points.

## 🔍 Overview

The core challenge addressed is identifying data that significantly deviates from the training (In-Distribution, ID) data. This work investigates two levels of granularity:
1.  **Subsample-wise Detection:** Determining if an entire *sample* (a collection of points) is OOD.
2.  **Point-wise Detection:** Determining if an individual *data point* is OOD.

## 🧠 Implemented Approaches

### 1. Subsample-wise Detection (AE + KS-Test)
This approach aims to label an entire data sample as $1$ (OOD) or $0$ (ID).

**Pipeline:**
*   **Autoencoder (AE):** An autoencoder is trained to reconstruct ID data. Given an input vector $x$, it produces a reconstruction $\hat{x}$.
*   **Reconstruction Error:** A scalar error is computed for a sample by aggregating the per-point errors. Specifically, the error for a dataset is derived from the root mean square error (RMSE) across its points and features:

    $$\text{error} = \sqrt{\frac{\sum_{i=1}^{d} (x_i - \hat{x}_i)^2}{d}}$$  
    where $d$ is the number of features.
*   **Statistical Test:** The Kolmogorov-Smirnov (KS) test is used to compare the distribution of reconstruction errors from a test sample against the distribution from a held-out ID (control) sample. A significantly different distribution (p-value < $\alpha$) indicates an OOD sample.

### 2. Point-wise Detection (AE + Distance-Based Classifier)
This approach aims to label each individual data point as $1$ (OOD) or $0$ (ID).

**Pipeline:**
*   **Autoencoder (AE):** The same trained autoencoder is used.
*   **Error Vector:** Instead of aggregating to a scalar, the full reconstruction error vector $(x - \hat{x})$ is preserved for each data point.
*   **Distance Metrics:** Various distance or metric scores are computed from this error vector to create a feature representation for classification. These include:
    *   Norm-based distances: ℓ₁, ℓ₂, ℓ∞.
    *   Statistical counts: `count_above_std`, `count_above_2std`, `count_above_3std`.
    *   Mahalanobis distance.
*   **Classifier:** A standard classifier (e.g., Decision Tree, Random Forest) is trained on these computed distances, using ID/OOD labels to predict the OOD status of new, individual points.

## 📝 Repository Structure

| File / Folder                  | Description                                                                                                                                                                                                  |
| :----------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `autoencode_real_data.ipynb`   | Main notebook containing experiments for both subsample-wise and point-wise approaches across 7 different datasets.                                                                                          |
| `model.py`                     | Contains the core `Autoencoder` class used for learning data representations and reconstruction.                                                                                                             |
| `pipeline.py`                  | Core functions for training, testing the **subsample-wise** approach, and running Monte Carlo simulations to evaluate method correctness (False Positive Rate control) and sensitivity (True Positive Rate). |
| `point_wise_classification.py` | Functions for the **point-wise** approach: calculating distances on error vectors, training classifiers, and generating synthetic OOD data by adding Gaussian noise to ID data.                              |
| `AE_ROC_modeling.py`           | Functions for modeling the transition from subsample-wise to point-wise detection via threshold calibration.                                                                                                 |
| `data/`                        | Directory containing 7 tabular datasets in CSV format provided by [ITMO NSS LAB](https://github.com/ITMO-NSS-team/OOD_Tab_Evaluation/tree/main).                                                                                                                              |
| `images/decision_tree/`        | Visualizations of decision trees obtained from the point-wise classifier, aiding in interpretability.                                                                                                        |

## 📊 Datasets

The research is conducted on 7 real-world tabular datasets with clear ID/OOD splits.

| Dataset           | Number of ID Samples | Number of OOD Samples | Number of Features | Features (Numerical / Categorical) |
| :---------------- | :------------------: | :-------------------: | :----------------: | :--------------------------------- |
| **Taxi**          |        10,000        |        10,000         |         7          | N: 7                               |
| **Electricity**   |        9,986         |        10,014         |         6          | N: 6                               |
| **Income**        |        20,380        |         9,782         |         12         | N: 4, C: 8                         |
| **MV X6**         |        20,384        |        20,384         |         9          | N: 6, C: 3                         |
| **Diabites**      |        34,288        |         1,500         |        183         | N: 10, C: 173                      |
| **California**    |        10,315        |        10,319         |         7          | N: 6, C: 1                         |
| **ACS Accidents** |        22,653        |         3,955         |         45         | N: 45                              |
