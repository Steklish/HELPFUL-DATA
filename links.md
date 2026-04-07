- Colab for yolo model training: `https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb`
- Pytorch lib `https://pytorch.org/get-started/locally/`
    - Example `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- SKLearn basic example `https://github.com/MariyaSha/scikit_learn_simplified`

    1. **Linear Regression** – Simple linear relationships, baseline model.  
        - `sklearn.linear_model.LinearRegression`

    2. **Polynomial Regression** – Curved, non‑linear trends (polynomial features + linear fit).  
        - `sklearn.preprocessing.PolynomialFeatures` + `sklearn.linear_model.LinearRegression`

    3. **Ridge Regression (L2)** – Ridge regularization for many correlated features.  
        - `sklearn.linear_model.Ridge`

    4. **Lasso Regression (L1)** – Feature selection; sparse models.  
        - `sklearn.linear_model.Lasso`

    5. **Elastic Net Regression** – Mix of L1 and L2 regularization.  
        - `sklearn.linear_model.ElasticNet`

    6. **Decision Tree Regression** – Non‑linear, rule‑based splits.  
        - `sklearn.tree.DecisionTreeRegressor`

    7. **Random Forest Regression** – Ensemble of trees, robust and accurate.  
        - `sklearn.ensemble.RandomForestRegressor`

    8. **Gradient Boosting Regression** – Sequential boosting (XGBoost‑style).  
        - `sklearn.ensemble.GradientBoostingRegressor`  
    (also `xgboost.XGBRegressor`, `lightgbm.LGBMRegressor`, etc., via separate packages)

    9. **Support Vector Regression (SVR)** – Margin‑based, works well in high dimensions.  
        - `sklearn.svm.SVR`

    10. **Gaussian Process Regression (GPR)** – Bayesian, uncertainty‑aware regression.  
        - `sklearn.gaussian_process.GaussianProcessRegressor`

    11. **RANSAC Regression** – Robust fitting ignoring outliers.  
        - `sklearn.linear_model.RANSACRegressor`

    12. **Neural Network Regressor** – Deep or shallow networks for complex patterns.  
        - `sklearn.neural_network.MLPRegressor`  
        (also PyTorch/TensorFlow models with custom wrappers)
