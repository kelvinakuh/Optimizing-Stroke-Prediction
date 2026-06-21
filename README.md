## 1\. Project Overview

This project implements an end-to-end machine learning pipeline to predict stroke risk using health and demographic data. The system uses an **XGBoost Classifier** optimized via **Bayesian Optimization** to prioritise the identification of at-risk individuals (Recall/Sensitivity).

## 2\. Software Requirements & Dependencies

To run this project, you will need **Python 3.x** installed along with the following libraries:

* **Data Manipulation**: pandas, numpy  
* **Visualisation**: matplotlib, seaborn  
* **Machine Learning**: scikit-learn, xgboost  
* **Class Imbalance**: imbalanced-learn (for SMOTE)  
* **Optimization**: scikit-optimize (for Bayesian Tuning)  
* **Explainability**: shap   
* **Hardware Check**: tensorflow (optional, used for GPU verification)

You can install these via pip:

pip install pandas matplotlib seaborn numpy scikit-learn xgboost imbalanced-learn scikit-optimize shap

## 3\. Instructions to Reproduce Results

* **Prepare Data**: Ensure train.csv and test.csv are placed in the /content/Work\_Book/ directory (or update the file paths in the script).  
* **Environment**: Run the script in an environment with Jupyter support (e.g., Google Colab or PyCharm with a Python 3.x interpreter).  
* **Execute Script**: Run the code in the sequence provided. The script will:  
  * Preprocess the data and handle class imbalance.  
  * Tune the model using Randomized Search and Bayesian Optimization.  
  * Generate SHAP plots for interpretability.  
  * Output final predictions to test\_predictions.csv.  
  * Generate a summary technical report in Report.txt.

## 4\. Execution Workflow

The pipeline follows these sequential steps:

* **Data Loading**: Ingest training and testing datasets using pandas.  
* **Preprocessing**:  
  * **Imputation**: Convert 'Unknown' entries in smoking\_status to None and impute using the mode.  
  * **Outlier Treatment**: Cap numerical features at 1st and 99th percentiles.  
  * **Encoding**: Apply One-Hot Encoding to categorical variables.  
  * **Clean-up**: Remove non-predictive id columns and handle duplicates.  
* **Resampling**: Apply **SMOTE** to balance the 'stroke' target variable.  
* **Splitting**: Perform an 80/20 stratified train-test split.  
* **Model Training & Optimization**:  
  * Establish a **Baseline** XGBoost model.  
  * Perform **RandomizedSearchCV** to narrow the hyperparameter range.  
  * Refine parameters using **Bayesian Optimization** (gp\_minimize).  
* **Interpretability**: Generate SHAP value plots to identify key predictors like age and glucose levels.  
* **Drift Detection**: Use Kolmogorov-Smirnov (KS) tests to ensure consistency between training and test distributions.  
* **Inference**: Generate predictions for test.csv and save to a CSV file.
