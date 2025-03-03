# CMO-NNR
# README

This project implements a **Distributionally Robust Optimization (DRO)** based training framework to improve model robustness under noisy and imbalanced datasets. The method supports various DRO algorithms and flexible strategies for updating learnable parameters.

## Parameter Descriptions

- **`DRO`**  
  Enable Distributionally Robust Optimization (DRO).  
  **Default:** `True`

- **`Q`**  
  Enable DRO with learnable parameters (`Q`).  
  **Default:** `True`

- **`DRO_method`**  
  Specify the DRO method.  
  **Options:** `[None, cvar, chisq, cvar_doro, chisq_doro, cvar_group]`  
  **Default:** `None`

- **`Q_update`**  
  Define the update strategy for the `Q` parameter.  
  **Options:** `[None, distance, variant, group, gradient]`  
  **Default:** `'distance'`

- **`D_k`**  
  Adjust the derived method.  
  `1` for **CMO-KL**, `2` for **CMO-Chi**.  
  **Default:** `2`

- **`file_name`**  
  The file name for saving experimental results.  
  **Example:** `'experiment_class6_noise0.15.csv'`  
  **Default:** `'experiment_class6_noise0.15.csv'`

- **`Distance_loss`**  
  Cross-environment feature distance constraint (denoted as **Γ** in the paper).  
  Controls the regularization strength on distribution differences.  
  **Default:** `0.0`

- **`Lambda`**  
  The first Lagrange multiplier controlling the weight of the DRO regularization term.  
  **Default:** `0.1`

- **`Miu`**  
  The second Lagrange multiplier controlling the weight of the cross-environment distance regularization.  
  **Default:** `0.1`

- **`P_lr`**  
  Learning rate for the `Q` parameter.  
  **Default:** `3e-5`
