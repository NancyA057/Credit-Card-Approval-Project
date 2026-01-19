# Introduction
This project analyzes credit card approval prediction using select machine learning models.
The goal is to explore applicant characteristics and build predictive models that estimates credit approval outcomes.

### Project Objective
- The primary objective of this project is to compare the performance of several machine learning models, and then identifying the most effective approach for predicting credit card approval outcomes.
- Project emphasis will be placed on model evaluation and selection rather than achieving maximum accuracy.

### Project Summary
- This project evaluates several machine learning models to determine the most effective approach for predicting credit card application approval outcomes.
Using applicant demographic information (`application_record.csv`) and historical credit behavior data (`credit_record.csv`), the analysis includes data preprocessing, exploratory data analysis, and
systematic model comparison. Several supervised learning models were employed, including Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and XGBoost.
Hyperparameter tuning for the K-Nearest Neighbors model was performed using GridSearchCV and RandomizedSearchCV to optimize performance. The results shows that, even after tuning, KNN was not the most suitable model for this task, while tree-based and ensemble methods demonstrated stronger predictive performance.
These findings support informed model selection for further analysis and potential real-world applications in credit risk assessment.

### Data
- Data was obtained from Kaggle, however, the dataset is too large to be hosted on Github. 
- The data consists of two files namely:
    - **application_record.csv**
    - **credit_record.csv**
- Data can be downloaded from the following link:
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction 

### Files
- **MLapp_proj.ipynb** - jupyter notebook containing detailed analysis - data preprocessing, exploratory data analysis and model development
- **application_record.csv** *(download from source link above)* - contains features such as annual income, educational level, marital status
- **credit_record.csv** *(download from source link above)* - contains features month_ balance, and status(number of days past due)

### Tools and Libraries
- Python
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost
- GridSearchCV
- RandomizedSearchCV

### How to Run
1. Download the dataset from the Kaggle link provided above
2. Place both CSV files in the project directory
3. Open `MLapp_proj.ipynb`
4. Run all cells

### Future Work
- Due to computational limitations, only a subset of the dataset was used in this project.
- Leveraging a cloud-based environment would enable analysis on the full dataset and allow for further evaluation of model performance.
