## Optimal Classification Tree (OCT)

### Overview

Decision trees are widely utilized machine learning tools for addressing classification problems. Given a dataset $X$ with $p$ features and corresponding labels $Y$ with $K$ distinct classes, decision trees typically operate by recursively applying binary tests at each internal node. These tests route data points through branches based on feature thresholds, ultimately classifying them at leaf nodes.

Common decision tree-based methods with high predictive accuracy, such as Random Forest and XGBoost, generally sacrifice interpretability, making it challenging to discern which features significantly influence decisions. The Classification and Regression Tree (CART) method is a widely used interpretable tree algorithm; however, CART employs greedy, top-down heuristics that can result in suboptimal predictive performance.

The Optimal Classification Tree (OCT) approach, introduced by Bertsimas and Dunn (2017) $1$, presents a novel alternative. OCT formulates the entire decision tree construction as a single mixed-integer programming (MIP) problem, enabling the generation of interpretable decision trees with optimal predictive performance.

### Project Description

This project, a collaborative effort by Xiaoyan Lin and Zhuoqiao Ouyang, aims to replicate and understand the OCT methodology through numerical experimentation. We implemented an OCT class in Python leveraging the Gurobi optimization solver. The implementation includes:

* **OCT fitting model**: Formulated as an MIP consistent with the original OCT paper. The model employs a CART-generated warmstart by default, with the flexibility to accept custom warmstart values.
* **Prediction function**: Outputs predictive accuracy on testing datasets.
* **Additional utilities**: Data extraction and preprocessing functions tailored for datasets sourced from the UCI Machine Learning Repository, cross-validation, and visualization of the OCT tree structure.

### Numerical Experiments

We evaluated our implementation using 5 real-world datasets. Each dataset was partitioned into 5 folds, comprising a training set (50%), validation set (25%), and testing set (25%). Given limited computational power, we restricted experiments to datasets with smaller sizes and capped the maximum depth of the OCT at 2.

Detailed experimental procedures, results, Python code, and usage guidelines are available in the accompanying project report provided in the repository.

### References

$1$ Bertsimas, D., & Dunn, J. (2017). Optimal Classification Trees. *Machine Learning, 106*(7), 1039–1082. $arXiv:2103.15965$

For more information on the Gurobi optimization solver: [Gurobi Optimization](https://www.gurobi.com).
