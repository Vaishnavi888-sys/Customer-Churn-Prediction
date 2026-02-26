# Customer-Churn-Prediction
##  Business Problem
Customer churn significantly impacts revenue for subscription-based businesses. Acquiring new customers costs 5–7x more than retaining existing ones.
This project builds a machine learning model to identify customers likely to churn, enabling proactive retention strategies and data-driven decision-making.

## Objective
Develop and evaluate classification models to predict customer churn using structured customer data, and select the best-performing model through hyperparameter tuning and cross-validation.

## Key Highlights
Built a complete end-to-end ML pipeline
Compared multiple tree-based models
Identified and addressed overfitting
Applied hyperparameter tuning using GridSearchCV
Achieved 75.09% test accuracy
Saved production-ready model using pickle

## Tech Stack
Programming: Python
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn
Model Persistence: Pickle

## Dataset Overview
The dataset contains customer-level information including:
Demographics
Account tenure
Subscription services
Billing information
Target variable: Churn (Yes/No)
This problem is framed as a supervised binary classification task.

## Machine Learning Workflow
1️) Data Preprocessing

. Handled missing values
. Encoded categorical variables
. Removed irrelevant features
. Prepared clean training dataset

2) Exploratory Data Analysis (EDA)
. Analyzed churn distribution
. Identified feature correlations
. Evaluated patterns influencing churn behavior
 
3) Model Development
. Implemented and compared:
    - Decision Tree Classifier
    - Random Forest Classifier
4) Model Performance Comparison
Model	Training Accuracy	Testing Accuracy
Decision Tree	100%	70.75%
Random Forest	100%	74.36%

Both models showed signs of overfitting (perfect training accuracy).
Random Forest demonstrated better generalization performance.

5) Hyperparameter Tuning
Performed GridSearchCV on Random Forest to improve generalization.

### Tuned Parameters:
. n_estimators
. max_depth
. min_samples_split
. min_samples_leaf
. Best Parameters:

  'max_depth': None,
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 200
  ----------------------
Best Cross-Validation Score:
0.7552
6) Final Model Performance
Final Test Accuracy: 75.09%
Improved generalization compared to base models
Selected Random Forest as final production model
7) Model Deployment Readiness
The trained model was serialized using Pickle:
churn_model.pkl
This enables:
. Reuse without retraining
. Integration into APIs or web applications
7) Deployment into production systems
 Project Structure
Customer-Churn-Prediction/
│
├── churn_prediction.ipynb
├── churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore

## Key Skills Demonstrated
. Supervised Machine Learning
. Model Evaluation & Overfitting Analysis
. Cross-Validation
. Hyperparameter Optimization
. Feature Handling
. Model Serialization

# Business Impact
This model provides a foundation for:
. Targeted retention campaigns
. Revenue preservation strategies
. Data-driven customer engagement
. Proactive churn prevention
Even a small improvement in churn prediction can translate into significant long-term revenue gains.
