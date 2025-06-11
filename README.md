# HR Analytics Full-Stack Portfolio Project: Predicting Employee Attrition for Next Quarter

## Project Background

This project delivers a robust, full-stack solution for understanding and predicting employee attrition, mimicking a real-world data science pipeline. The primary goal is to leverage modern machine learning models (XGBoost, Random Forest, and Logistic Regression) to accurately determine the likelihood of an employee's termination within the next quarter (90 days).

The initiative stems from identified limitations in an existing `risk_of_exit_score`, which consistently showed inflated attrition probabilities (up to 90%) for employees who remained with the company. This suggested the score was based on an outdated dataset, failing to reflect current workforce dynamics. By implementing contemporary machine learning techniques, this project aims to provide more precise, actionable insights into attrition risk, empowering HR stakeholders with data-driven decision-making capabilities.

### Project Pipeline Overview

This project embodies a comprehensive data science workflow:

-   **Data Transformation (SSMS):** Utilized SQL Server Management Studio (SSMS) for robust data cleaning, preprocessing, and feature engineering, transforming raw employee snapshots and leave requests into a structured dataset suitable for machine learning.
-   **Advanced Analytics (Python):** Employed Python (via Jupyter Notebook) for advanced analytics, including the implementation and evaluation of XGBoost, Random Forest, and Logistic Regression models. Key techniques like SHAP (SHapley Additive exPlanations) were used for model interpretability, and SMOTE (Synthetic Minority Over-sampling Technique) addressed class imbalance in the attrition dataset.
-   **Insights & Visualization (Tableau):** Developed interactive Tableau dashboards to support data exploration and provide professional, actionable insights for HR business partners. The dashboards highlight the improved prediction power of the new models compared to the old `risk_of_exit_score` and offer transparency on their accuracy.

**Key Technologies Used:** SSMS, Python (Pandas, Scikit-learn, Imbalanced-learn (SMOTE), SHAP, Pickle), Jupyter Notebook, Tableau.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

## Data Structure

The raw data comprises monthly snapshots of individual employees (`snapshots_updated.csv`) and their corresponding leave requests (`leave_requests.csv`). Each snapshot provides a comprehensive view of an employee's status at a specific point in time.

![Excel Table](https://github.com/user-attachments/assets/6e82d5d1-23db-4962-b0cf-8d5c5c0b81bb)

The raw data is located in `data/raw/snapshots_updated.csv` and `data/raw/leave_requests.csv`. After SQL processing, the cleaned and transformed data is saved as `data/cleaned/full summary.csv`. This cleaned dataset includes aggregated leave requests, crucial flags (`target_variable`, `ever_terminated_flag`), and excludes employees with fewer than 6 monthly snapshots to ensure sufficient historical context for robust modeling.

## Executive Summary

### Overview:

This project delivers a comprehensive solution for understanding and predicting employee attrition. Key insights reveal that attrition frequently peaks between 12-36 months of tenure, with specific departments and employees exhibiting certain performance/engagement profiles showing higher risk. The newly implemented machine learning models demonstrate a significant improvement over the existing `risk_of_exit_score` by providing more precise predictions and actionable insights into the underlying drivers of attrition. The interactive Tableau dashboards empower HR business partners to proactively identify at-risk employees and assess overall workforce health, enabling targeted retention strategies.

Link to Dashboards: ![Tableau]https://public.tableau.com/shared/GBZ4PPGQD?:display_count=n&:origin=viz_share_link

## Data Preparation with SQL (SSMS)

The initial raw datasets (`snapshots_updated.csv` and `leave_requests.csv`) were ingested into an SSMS database. The `Fixed Query.sql` script then executed a series of critical transformations to prepare the data for machine learning.

Key steps performed in the SQL script:

-   `Employee Snapshot Counts` **CTE:** Calculated the number of distinct monthly snapshots for each employee, along with their first, last, and actual `termination_date`. It filtered out employees with fewer than 6 monthly snapshots to ensure adequate historical context.
-   `QualifyingSnapshots` **CTE:** Selected all monthly snapshots for employees identified as having sufficient history from the `EmployeeSnapshotCounts` CTE.
-   `EmployeeLeaveSummaryPerSnapshot` **CTE:** Aggregated `leave_requests.csv` data by employee and `snapshot_date`, summarizing different leave types and approval statuses.
-   **Final Join & Feature Creation:** Joined these CTEs to construct the `full_summary.csv` dataset. Crucial new features were created, including:
    -   `target_variable`: The binary target variable, indicating if an employee terminated within 90 days (next quarter) after a specific `snapshot_date`.
    -   `ever_terminated_flag`: A flag indicating if an employee ever terminated in the dataset's history, primarily used for filtering active employees in the current snapshot.
    -   `termination_date`: The actual date of termination for reference.

## Code Implementation (Python via Jupyter Notebook)

This section provides a high-level overview of the technical implementation and the Python code's functionality within the `Full Working Code_v1.ipynb` Jupyter Notebook.

### Python Libraries Used:

-   `pandas`: For robust data loading, manipulation, cleaning, and feature engineering (e.g., handling dates, creating `months_since_hire`).
-   `numpy`: For efficient numerical operations.
-   `sklearn` (Scikit-learn): Comprehensive library for machine learning, including:
    -   `model_selection`: `train_test_split`, `cross_val_score` for data splitting and cross-validation.
    -   `preprocessing`: `StandardScaler`, `OneHotEncoder` for numerical scaling and categorical encoding.
    -   `compose`: `ColumnTransformer` for applying different transformations to different columns.
    -   `impute`: `SimpleImputer` for handling missing values.
    -   `pipeline`: `Pipeline` for streamlining preprocessing and modeling steps.
    -   `ensemble`: `RandomForestClassifier`.
    -   `linear_model`: `LogisticRegression`.
    -   `metrics`: `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve` for model evaluation.
-   `imblearn.over_sampling.SMOTE`: For Synthetic Minority Over-sampling Technique, addressing class imbalance by generating synthetic samples for the minority class (`future_terminated_flag=1`).
-   `xgboost`: `XGBClassifier` for gradient boosting models.
-   `shap`: For SHapley Additive exPlanations, providing local and global model interpretability.
-   `matplotlib.pyplot` and `seaborn`: For creating various data visualizations, including distribution plots, correlation matrices, and model performance graphs.
-   `pickle`: For serializing and deserializing Python objects, used to save and load trained machine learning models and preprocessors.

### Model Selection Rationale:

For this HR attrition prediction project, a strategic choice of machine learning models was made, prioritizing a balance of interpretability, robustness, and high predictive performance typical for tabular binary classification problems.

-   **Logistic Regression (Interpretable Baseline):** Chosen as the foundational linear model. Its interpretability, allowing for direct understanding of feature impacts on attrition likelihood, is invaluable for HR stakeholders. It also provides a strong, efficient baseline for comparison against more complex models.
-   **Random Forest (Robust Ensemble):** This ensemble method was selected for its ability to capture complex, non-linear relationships and interactions between features without extensive manual engineering. Its ensemble nature significantly reduces the risk of overfitting, making it a robust choice, and it naturally provides insights into feature importance.
-   **XGBoost (High-Performance Powerhouse):** As a state-of-the-art gradient boosting framework, XGBoost was included for its exceptional predictive performance and efficiency. It excels at handling large datasets and complex relationships, often delivering industry-leading accuracy in tabular data competitions. Its built-in regularization also helps prevent overfitting.

**Why Other Common Models Were Not Prioritised:**

While many other machine learning algorithms exist, several were not the primary focus for this project due to specific trade-offs relative to the problem's requirements:

-   **Individual Decision Trees:** Prone to severe overfitting, which is effectively mitigated by ensemble methods like Random Forest and XGBoost.
-   **Support Vector Machines (SVMs):** Can be computationally expensive on larger datasets and sensitive to feature scaling. Interpretability, especially with non-linear kernels, is also a significant challenge for business understanding.
-   **K-Nearest Neighbors (KNN):** Computationally intensive for prediction on substantial datasets, sensitive to feature scaling, and can suffer from the "curse of dimensionality" in higher-dimensional feature spaces.
-   **Naive Bayes:** Relies on a strong assumption of feature independence, which is rarely true in correlated HR datasets, often leading to suboptimal performance compared to tree-based models.
-   **Deep Learning / Neural Networks:** While powerful, they are typically overkill and less interpretable for structured tabular data of this nature compared to tree-based models, often requiring more data and computational resources to achieve comparable results. The priority for HR insights leans towards interpretability, making simpler yet powerful models more suitable.

### Code Structure Overview:

The Python script in the Jupyter Notebook is meticulously structured to guide the analysis from data ingestion to model training, evaluation, and insight generation:

1.  **Environment Setup & Imports:** Consolidates all necessary library imports and defines a global `overwrite_files` control flag.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

2.  **Data Loading:** Loads the `full summary.csv` dataset into a pandas DataFrame and provides initial data overview.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

3.  **Exploratory Data Analysis (EDA):** Performs in-depth analysis of the dataset, including distribution plots, correlation matrices, and initial insights into `ever_terminated_flag`.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

4.  **Feature Engineering & Target Variable Creation:** Converts date columns, sorts data by employee, and crucially generates the `target_variable` (termination within 90 days) as the target variable. It also creates features like `months_since_hire`.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

5.  **Data Splitting & Preprocessing Pipeline:** Defines features (`X`) and target (`y`), splits data into training and testing sets (80% train, 20% test, stratified), and constructs robust preprocessing pipelines for numerical (imputation, scaling) and categorical (imputation, one-hot encoding) features.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

6.  **Handling Imbalance with SMOTE:** Applies SMOTE to the training data to address class imbalance, ensuring models are not biased towards the majority class (non-terminated employees).

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

7.  **Model Training & Cross-Validation:** Initializes, trains, and performs 5-fold cross-validation on `RandomForestClassifier`, `LogisticRegression`, and `XGBClassifier` models using the preprocessed and resampled training data.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

8.  **Model Evaluation on Test Set:** Evaluates each trained model on the unseen test set, providing detailed `classification_report`s, `confusion_matrix` visualizations, and `ROC curves` with AUC scores.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

9.  **Model Interpretability (SHAP Values):** Computes and visualizes SHAP values for the XGBoost model to explain global feature importance and individual prediction contributions, enhancing transparency.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

10. **Feature Importance & Model Coefficients:** Extracts and saves feature importance scores (for tree-based models) and coefficients (for Logistic Regression) to CSV files for further analysis.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

11. **Artifact Saving:** Saves the trained `preprocessor.pkl`, resampled training data CSVs, transformed test data CSVs, and the comprehensive `main_for_tableau.csv` (full original test set with all predictions).

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

12. **Current Snapshot Generation for BI/Dashboarding:** Processes the `main_for_tableau.csv` to extract and save the `current_snapshot.csv`, containing only the latest, relevant data for truly active employees (those not terminated and with recent snapshot dates), ready for current insights and reporting.

![Flowchart](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a)

### Key ML Implementation Details:

-   **Train-Test Split:** Dataset split into 80% training and 20% testing sets, stratified by the `target_variable` to maintain class distribution.
-   **Cross-Validation:** K-Fold cross-validation (5-fold) employed during model training to enhance generalization and reduce overfitting.
-   **Hyperparameter Tuning:** Basic hyperparameter tuning was performed for each model to optimize their performance on this dataset.
-   **Model Evaluation Metrics:** Performance was assessed using critical metrics for imbalanced classification: Accuracy, Precision, Recall, F1-Score, ROC AUC Score, and Confusion Matrix.
-   **Correlation Matrix and Termination Spread:** Initial EDA included generating a correlation matrix to understand feature relationships and analyzing employee tenure distributions for terminated individuals to highlight common attrition periods.

## Tableau Dashboards & Interactive Insights

Three interactive Tableau dashboards were developed to provide HR stakeholders with comprehensive, actionable insights:

### Dashboard 1: Workforce Health & Attrition Overview

**Purpose:** Provides a high-level overview of the company's workforce health, key HR metrics, and overall attrition trends, enabling executives and HR leaders to quickly grasp the current state.

![Dashboard 1 Screenshot](https://github.com/user-attachments/assets/d0b63d8f-4f04-4268-9303-3cccb6beb45a) **Key Insights:**

-   Identified 'Marketing' and 'HR' departments as having historically higher annualized attrition rates compared to other departments.
-   Overall company attrition has shown a consistent upward trend over recent quarters, signaling a need for proactive intervention.
-   The "Predicted Attrition (Next Quarter)" shows X employees currently at high risk, providing a clear target for proactive HR engagement.

### Dashboard 2: Current Attrition Risk & Intervention

**Purpose:** Focuses on the output of the predictive models, highlighting employees currently at high risk of attrition and identifying the primary factors contributing to their risk. Designed to support HR business partners in targeted interventions.

![Dashboard 2 Screenshot](https://github.com/user-attachments/assets/6e82d5d1-23db-4962-b0cf-8d5c5c0b81bb) **Key Insights:**

-   The SHAP plot clearly demonstrates that `months_since_hire` and `performance_rating` are top drivers of attrition risk, with lower values in these features correlating with higher risk.
-   A total of Y employees are currently flagged as high-risk by the models, primarily concentrated in specific job titles and locations, enabling focused support.
-   The model identifies specific cohorts (e.g., those with low engagement and recent negative performance reviews) as having significantly elevated attrition probability.

### Dashboard 3: Historical Attrition Drivers & Model Performance

**Purpose:** Dives deeper into the historical factors driving attrition and rigorously evaluates the performance of the machine learning models. Provides transparency on model accuracy and the reliability of predictions.

![Dashboard 3 Screenshot](https://github.com/user-attachments/assets/6f9717d5-f9c3-426e-8f7b-8053a51924c8) 

**Key Insights:**

-   The Confusion Matrix reveals that while the models excel at predicting employees who will stay (high specificity), their ability to correctly identify employees who will leave (recall for the minority class) is a key area for potential improvement for certain models.
-   Analysis of historical terminations shows that the highest volume of attrition occurs between 12-36 months of employment, suggesting a critical retention window.
-   Comparison with the old `risk_of_exit_score` demonstrates that the new ML models provide a significantly more accurate and balanced prediction of attrition, validated by higher F1-scores and improved ROC AUC.

## Recommendations

Based on the analysis and model insights presented in the Tableau dashboards, the following actionable recommendations are put forth for HR stakeholders:

-   **Targeted Retention Programs for Mid-Tenure Employees:** Implement specific engagement or development programs for employees approaching their 12-36 month tenure mark, as this was consistently identified as a high-risk period for attrition across various departments.
-   **Proactive Intervention for High-Risk Individuals:** Utilize the "Current Attrition Risk & Intervention" Tableau dashboard to regularly identify and reach out to employees flagged with high predicted attrition risk, particularly those in departments with historically high churn rates.
-   **Review Performance Management & Engagement Strategies:** Investigate the relationship between lower performance ratings and engagement scores with attrition, especially in high-risk groups. Develop targeted support or improvement plans to address these areas.
-   **Data Quality Improvement:** Establish protocols for regular data audits or explore integrating HRIS with analytics platforms to ensure accurate and up-to-date employee status information, specifically addressing the issue of stale snapshots without termination flags.

## Limitations

This project provides strong predictive capabilities and actionable insights but comes with certain inherent limitations:

-   **Data Granularity & Features:** The analysis is constrained by the features available in the provided simulated dataset. Incorporating additional real-world data (e.g., manager changes, team-level dynamics, compensation review history, internal mobility, employee feedback scores, specific project assignments) could further enhance model accuracy and provide deeper, more nuanced insights into attrition drivers.
-   **Model Interpretability Trade-offs:** While SHAP values significantly improve the interpretability of complex machine learning models like XGBoost, some level of "black box" behavior remains. Effective application of insights requires strong business context and collaboration with HR domain experts.
-   **Generalizability & Simulation Bias:** The models are trained and validated on a simulated dataset. Their performance and the specificity of insights may vary when applied to a real-world company's unique workforce dynamics and historical data without further retraining or fine-tuning on actual enterprise data.
-   **Lagging Indicator:** The `target_variable` predicts attrition 90 days (one quarter) out from the snapshot date. While valuable for proactive planning, even earlier indicators or more frequent (e.g., weekly) data snapshots could enable even more immediate interventions.
-   **Pre-existing Data Quality Gaps:** As identified during the data preparation phase, some employee snapshot entries were outdated without corresponding termination flags. While addressed for the `current_snapshot.csv`, such gaps in raw data could subtly influence overall historical attrition rate calculations if not thoroughly managed.