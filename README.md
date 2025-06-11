![image](https://github.com/user-attachments/assets/7ca6b358-57a4-404a-bd86-b81c0cec5a8d)
# HR Analytics Full-Stack Portfolio Project

## Project Background

This project delivers a robust, full-stack solution for understanding and predicting employee attrition, mimicking a real-world data science pipeline. The primary goal is to leverage modern machine learning models (XGBoost, Random Forest, and Logistic Regression) to accurately determine the likelihood of an employee's termination within the next quarter (90 days).

The initiative stems from identified limitations in an existing `risk_of_exit_score`, which consistently showed inflated attrition probabilities (up to 90%) for employees who remained with the company. This suggested the score was based on an outdated dataset, failing to reflect current workforce dynamics. By implementing contemporary machine learning techniques, this project aims to provide more precise, actionable insights into attrition risk, empowering HR stakeholders with data-driven decision-making capabilities.

### Project Pipeline Overview

This project embodies a comprehensive data science workflow:

-   **Data Transformation (SSMS):** Utilized SQL Server Management Studio (SSMS) for robust data cleaning, preprocessing, and feature engineering, transforming raw employee snapshots and leave requests into a structured dataset suitable for machine learning.
-   **Advanced Analytics (Python):** Employed Python (via Jupyter Notebook) for advanced analytics, including the implementation and evaluation of XGBoost, Random Forest, and Logistic Regression models. Key techniques like SHAP (SHapley Additive exPlanations) were used for model interpretability, and SMOTE (Synthetic Minority Over-sampling Technique) addressed class imbalance in the attrition dataset.
-   **Insights & Visualisation (Tableau):** Developed interactive Tableau dashboards to support data exploration and provide professional, actionable insights for HR business partners. The dashboards highlight the improved prediction power of the new models compared to the old `risk_of_exit_score` and offer transparency on their accuracy.

**Key Technologies Used:** SSMS, Python (Pandas, Scikit-learn, Imbalanced-learn (SMOTE), SHAP, Pickle), Jupyter Notebook, Tableau.

## Data Structure

The raw data comprises monthly snapshots of individual employees (`snapshots_updated.csv`) and their corresponding leave requests (`leave_requests.csv`). Each snapshot provides a comprehensive view of an employee's status at a specific point in time.

![Dataset Raw](https://github.com/user-attachments/assets/d4814690-1473-4513-98ee-30d10a76865f)

The raw data is located in `raw/snapshots_updated.csv` and `raw/leave_requests.csv`. After SQL processing, the cleaned and transformed data is saved as `sql/full summary.csv`. This cleaned dataset includes aggregated leave requests, crucial flags (`target_variable`, `ever_terminated_flag`), and excludes employees with fewer than 6 monthly snapshots to ensure sufficient historical context for robust modeling.

## Executive Summary

### Overview:

This project delivers a comprehensive solution for understanding and predicting employee attrition. Key insights reveal that attrition frequently peaks between 12-36 months of tenure, with specific departments and employees exhibiting certain performance/engagement profiles showing higher risk. The newly implemented machine learning models demonstrate a significant improvement over the existing `risk_of_exit_score` by providing more precise predictions and actionable insights into the underlying drivers of attrition. The interactive Tableau dashboards empower HR business partners to proactively identify at-risk employees and assess overall workforce health, enabling targeted retention strategies.

Link to Dashboards: [Tableau](https://public.tableau.com/shared/GBZ4PPGQD?:display_count=n&:origin=viz_share_link)

### Key Results:
- **Improved Prediction Accuracy**: 97.48% model accuracy vs. legacy system's inflated scores where individuals had values in the high 90s despite never leaving the company
- **Identified High-Risk Employees**: 10 employees (3.69%) flagged for immediate intervention
- **Departmental Insights**: Marketing and HR departments identified as highest attrition risk (>1.50%)
- **Business Impact**: Current measures implemented to prevent attrition is working as the change in average attrition rate was the lowest in 4 years (0.14%)

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

For this HR attrition prediction project, a strategic choice of machine learning models was made, prioritising a balance of interpretability, robustness, and high predictive performance typical for tabular binary classification problems.

-   **Logistic Regression (Interpretable Baseline):** Chosen as the foundational linear model. Its interpretability, allowing for direct understanding of feature impacts on attrition likelihood, is invaluable for HR stakeholders. It also provides a strong, efficient baseline for comparison against more complex models.

--- Classification Report ---

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.99      | 0.76   | 0.86     | 3568    |
| 1     | 0.05      | 0.58   | 0.09     | 79      |
| **Accuracy** |           |        | **0.76** | **3647**|
| Macro Avg | 0.52      | 0.67   | 0.48     | 3647    |
| Weighted Avg| 0.97      | 0.76   | 0.84     | 3647    |

<p>
    <img width="500" height="500" src="https://github.com/user-attachments/assets/bab5b810-1383-431e-a31e-971876c5c805" >
    <img width="500" height="500" src="https://github.com/user-attachments/assets/9ac9f10a-89ec-4412-95a4-a83e92e5bc0a" >
</p>

-   **Random Forest (Robust Ensemble):** This ensemble method was selected for its ability to capture complex, non-linear relationships and interactions between features without extensive manual engineering. Its ensemble nature significantly reduces the risk of overfitting, making it a robust choice, and it naturally provides insights into feature importance.

--- Classification Report ---

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.98      | 1.00   | 0.99     | 3568    |
| 1     | 0.58      | 0.23   | 0.33     | 79      |
| **Accuracy** |           |        | **0.98** | **3647**|
| Macro Avg | 0.78      | 0.61   | 0.66     | 3647    |
| Weighted Avg| 0.97      | 0.98   | 0.98     | 3647    |

<p>
    <img width="500" height="500" src="https://github.com/user-attachments/assets/d443ee56-fd40-4385-8043-29ef2bcd5728" >
    <img width="500" height="500" src="https://github.com/user-attachments/assets/229994ec-efec-41f2-b3b2-03d51339f054" >
</p>

-   **XGBoost (High-Performance Powerhouse):** As a state-of-the-art gradient boosting framework, XGBoost was included for its exceptional predictive performance and efficiency. It excels at handling large datasets and complex relationships, often delivering industry-leading accuracy in tabular data competitions. Its built-in regularization also helps prevent overfitting.

--- Classification Report ---

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.98      | 0.99   | 0.99     | 3568    |
| 1     | 0.33      | 0.16   | 0.22     | 79      |
| **Accuracy** |           |        | **0.97** | **3647**|
| Macro Avg | 0.66      | 0.58   | 0.60     | 3647    |
| Weighted Avg| 0.97      | 0.97   | 0.97     | 3647    |

<p>
    <img width="500" height="500" src="https://github.com/user-attachments/assets/7beade4e-0816-4585-9a01-7e3da3d9a6ae" >
    <img width="500" height="500" src="https://github.com/user-attachments/assets/30f6537b-117d-4699-8a83-9dcdec31780e" >
</p>


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

![image](https://github.com/user-attachments/assets/a4eff3cc-dc8d-4993-a79e-d2b047ae82fa)

2.  **Data Loading:** Loads the `full summary.csv` dataset into a pandas DataFrame and provides initial data overview.

![image](https://github.com/user-attachments/assets/4408e9e7-0ef5-4160-b3cc-06b0aff24845)

3.  **Exploratory Data Analysis (EDA):** Performs in-depth analysis of the dataset, including distribution plots, correlation matrices, and initial insights into `ever_terminated_flag`.

![image](https://github.com/user-attachments/assets/11844ec6-9ff7-4e96-a860-b560c52fdfd2)

4.  **Feature Engineering & Target Variable Creation:** Converts date columns, sorts data by employee, and crucially generates the `target_variable` (termination within 90 days) as the target variable. It also creates features like `months_since_hire`.

![image](https://github.com/user-attachments/assets/2a052886-5b44-4c58-9439-a93db52f5be2)

5.  **Data Splitting & Preprocessing Pipeline:** Defines features (`X`) and target (`y`), splits data into training and testing sets (80% train, 20% test, stratified), and constructs robust preprocessing pipelines for numerical (imputation, scaling) and categorical (imputation, one-hot encoding) features.

![image](https://github.com/user-attachments/assets/e60c9927-480e-44d1-9b45-262d309e8853)

6.  **Handling Imbalance with SMOTE:** Applies SMOTE to the training data to address class imbalance, ensuring models are not biased towards the majority class (non-terminated employees).

![image](https://github.com/user-attachments/assets/b1345c1b-9b19-49b1-9008-8465ecfa1bf3)

7.  **Model Training & Cross-Validation:** Initializes, trains, and performs 5-fold cross-validation on `RandomForestClassifier`, `LogisticRegression`, and `XGBClassifier` models using the preprocessed and resampled training data.

![image](https://github.com/user-attachments/assets/418d76e2-50c8-4bf0-98f7-d443d4bffe02)

8.  **Model Evaluation on Test Set:** Evaluates each trained model on the unseen test set, providing detailed `classification_report`s, `confusion_matrix` visualizations, and `ROC curves` with AUC scores.

![image](https://github.com/user-attachments/assets/7c21abcb-19b3-4e6f-8539-ff0b4c7ec1ca)

9.  **Model Interpretability (SHAP Values):** Computes and visualizes SHAP values for the XGBoost model to explain global feature importance and individual prediction contributions, enhancing transparency.

![image](https://github.com/user-attachments/assets/5ea51ba0-5854-4c75-91f7-d3f6646cd55d)

10. **Feature Importance & Model Coefficients:** Extracts and saves feature importance scores (for tree-based models) and coefficients (for Logistic Regression) to CSV files for further analysis.

![image](https://github.com/user-attachments/assets/087097c8-89f3-4c90-9393-578b0f5724fc)

11. **Artifact Saving:** Saves the trained `preprocessor.pkl`, resampled training data CSVs, transformed test data CSVs, and the comprehensive `main_for_tableau.csv` (full original test set with all predictions).

![image](https://github.com/user-attachments/assets/a5e4424d-ff93-4c42-a521-295704e9baf0)

12. **Current Snapshot Generation for BI/Dashboarding:** Processes the `main_for_tableau.csv` to extract and save the `current_snapshot.csv`, containing only the latest, relevant data for truly active employees (those not terminated and with recent snapshot dates), ready for current insights and reporting.

![image](https://github.com/user-attachments/assets/c638dcb9-479a-4228-a3bc-28881bd78ebf)

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

![Dashboard 1](https://github.com/user-attachments/assets/8fd57470-294f-423a-8201-1dfff4319b4b)

**Key Insights:**

- **Company Size:** 271 total active employees with 27.5 months average tenure
- **Attrition Rate:** Q2 2025 predicted at 1.35% attrtion rate based on new model 
- **At-Risk Population:** 10 employees (3.69%) identified as high attrition risk
- **Historical Trend:** Significant improvement from 2021-2024, with attrition rates remaining the same from 2023 (2.58%) to 2024 (2.72%)

### Dashboard 2: Current Attrition Risk & Intervention

**Purpose:** Focuses on the output of the predictive models, highlighting employees currently at high risk of attrition and identifying the primary factors contributing to their risk. Designed to support HR business partners in targeted interventions.

![Dashboard 2](https://github.com/user-attachments/assets/a04c1244-19f7-40b7-8104-74ebc3bef4e2)

**Key Insights:**

- **High-Risk Departments:**

    - **Engineering:** Highest attrition risk (longest bar in profile chart)
    - **Sales:** Second highest risk department
    - **Finance:** Moderate risk with 1.40% predicted Q2 attrition

- **Stable Departments:**

    - **Operations:** Lowest attrition risk (1.41% predicted)
    - **HR:** Relatively stable (1.84% predicted)

### Dashboard 3: Historical Attrition Drivers & Model Performance

**Purpose:** Dives deeper into the historical factors driving attrition and rigorously evaluates the performance of the machine learning models. Provides transparency on model accuracy and the reliability of predictions.

![Dashboard 3](https://github.com/user-attachments/assets/d47feb78-b54e-4eab-8d90-dcfee8eece40)

**Key Insights:**

- **Strengths:**

    - **High Accuracy:** 97.48% model accuracy demonstrates strong predictive capability
    - **Excellent Specificity:** Low false positive rate for retention predictions

- **Weaknesses:**

    - **Low Precision:** 33.33% precision indicates high false positive rate for attrition predictions
    - **Poor Recall:** 16.46% recall means the model misses many actual attrition cases
    - **Class Imbalance:** Model struggles with the rare event nature of attrition

- **Immediate Actions (Next 30 Days)**

    - **Focus on High-Risk Employees:** Implement retention strategies for the 10 identified high-risk employees
    - **Engineering Department Intervention:** Conduct stay interviews and address specific concerns in the engineering team
    - **Performance-Engagement Correlation:** Address the clear relationship between low engagement scores and high attrition risk

- **Strategic Initiatives (3-6 Months)**

    - **Tenure-Based Retention Programs:** Develop targeted programs for employees in their first 24 months (highest risk period)
    - **Department-Specific Strategies:** Create tailored retention approaches for Engineering and Sales teams
    - **Engagement Score Improvement:** Implement initiatives to raise engagement scores, particularly for employees scoring below 70
    - **Training and Development:** Address the "months since last training" factor that appears in the risk drivers

- **Model Improvement Recommendations**

    - **Threshold Optimization:** Adjust prediction thresholds to improve recall while maintaining acceptable precision
    - **Feature Engineering:** Explore additional variables that might improve prediction of actual attrition cases
    - **Ensemble Methods:** Consider combining multiple models to improve both precision and recall
    - **Temporal Analysis:** Incorporate seasonal patterns and business cycle effects

## Limitations

- **Data Limitations**

    - **Sample Size:** With only 18,000 rows of data from 755 unique employees, the data might be too biased for robust machine learning, especially for rare events
    - **Historical Bias:** 79 actual terminations from 2021-2025 may not provide sufficient positive examples for model training
    - **Feature Completeness:** Missing contextual factors (market conditions, compensation benchmarking, manager quality)
    - **Temporal Scope:** Analysis limited to 2021-2025 data which may not capture longer-term trends

- **Model Limitations**

    - **Class Imbalance:** Severe imbalance between retention and attrition cases leads to poor minority class prediction
    - **Precision-Recall Trade-off:** Current model optimises for accuracy rather than balanced performance on both classes
    - **Interpretability:** Some risk factors may be correlation rather than causation
    - **Dynamic Factors:** Model may not capture rapidly changing employee sentiments or external market conditions

- **Business Context Limitations**

    - **Industry Benchmarking:** No external benchmarks provided to assess if current rates are competitive
    - **Cost-Benefit Analysis:** ROI of retention interventions not quantified
    - **Voluntary vs. Involuntary:** No distinction between voluntary departures and terminations
    - **Exit Interview Data:** Lack of qualitative insights from departing employees

- **Operational Limitations**

    - **Real-time Updates:** Dashboard appears to be point-in-time analysis rather than real-time monitoring
    - **Intervention Tracking:** No mechanism shown to track success of retention efforts
    - **Manager-Level Insights:** Analysis aggregated at department level may miss team-specific issues
    - **Seasonal Adjustments:** No apparent adjustment for seasonal hiring/departure patterns
