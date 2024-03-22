# Heart Attack Prediction Project 

## Group Members
- Mahsa Nafei
- Jesús Hernández
- Fernando López
- Alexandru Arnautu

## Overview
Our repository combines two data sources and API Coordinates to predict heart attack risk using machine learning and visualization analysis. It features a Flask-based application for risk prediction, a data processing script using Postgresql in Amazon RDS, and employs Spark for data cleaning and preparation. We employed data Plotly, Pandas and Folium for visualization and analysis.
 
### Objective 
This project focuses on using a dataset to create a predictive model for assessing an individual's risk of having a heart attack. It aims to understand how specific  lifestyle factors influence the probability of heart risk

- Age
- Blood Pressure
- Cholesterol Levels 
- Diabetic Status
- Hours of Sleep
- Medication Use
- Obesity
- Physical Activity 
- Smoking 
- Triglycerides

Our project seeks to examine the data outlined above to help insurance companies provide  more detail specific health programs for patients affected by heart risk, by looking at related factors. A critical aspect of the project is ensuring fairness and reducing bias in predictions across different demographic groups, and instead focused on creating a global pattern rather than continent/country specific ones. Our goals throughout the making of this analysis, was to focus on  improving risk assessment accuracy through supervised machine learning to support informed decision-making in heart attack risk evaluation.


#### Questions
- Health Risk assessment
  - Question: Based on the available data, how effectively does the model capture and estimate an individual's likelihood of experiencing a heart attack?
  - Question: Which feature in our dataset impacts the possibility of a heart attack the most?

- Fairness and Bias:
  - Question: How does the analysis account for fairness and avoid bias in predicting heart attack risks across different demographic groups?
  - Question: What is the age distribution of the heart attack risk in the data set?
  - Question: What is the distribution by country and continent of heart attack risk?

- Impact of Lifestyle Factors on Risk Assessment:
  - Question: What is the relative contribution of lifestyle factors (e.g., smoking, obesity, physical activity, hours of sleep) versus clinical parameters (e.g., cholesterol levels, triglycerides, blood pressure) in predicting heart attack risk?
  - Question: How does the amount of sleep and physical activity influence the risk of an individual becoming susceptible to heart attack?
  - Question: How does triglycerides and cholesterol levels impact the risk of heart attack at a continental scale? 

#### Code Structure

### Data Preparing
- **preparing_data.ipynb** This file details the integration process of the heart attack CSV file sourced from Kaggle with data from the Countries Now API. Using Apache Spark, we conducted data cleaning and merging operations to adjust the datasets for further analysis.

### Endpoints
1. **heartattack prediction dataset**
 - URL: [https://aws-project-4.s3.ca-central-1.amazonaws.com/heart_attack_prediction_dataset.csv]
   - Description: Connection to the heart attack data set in an amazon S3 bucket.


### Data processing 
- **app.py:** The main Python script defining the Flask application, including routes for different pages and functionalities related to heart attack prediction.

- **traning_data.ipynb** This file documents our exploration of various machine-learning models for supervised  training on the dataset. We experimented with logistic regression, random forest models, KN means, and neural networks to enhance prediction accuracy. Through a combination of techniques, including data sampling with a random oversample, grid search, and employment of a random forest classifier, we achieved a prediction accuracy of 74%. This file contains the connection to the data base. 

#### Endpoints
1. **Deployment**
   - URL: [http://heartattackriskpredictorv2.us-east-1.elasticbeanstalk.com/](http://heartattackriskpredictorv2.us-east-1.elasticbeanstalk.com/)
   - Description: Displays the heart attack app for the user to input data.

or 

1. **Home Page**
   - URL: [http://localhost:5000/](http://localhost:5000/)
   - Description: Displays the heart attack app for the user to input data.
2. **results**
   - URL: [http://localhost:5000/predict](http://localhost:5000/)
   - Description: Displays the results of the heart attack application.


### Postgres SQL connection 
 - **Reesources/Data_base_setup/PostgreSQL database_Amazon RDS.ipynb** This file contains the information and functions for the connections with amazon RDS porstgres SQL data base. 
 - **Reesources/queries** This folder contains all the used queries to create the tables in the data base heart_attack_prediction_db.

### Endpoint
1. **heart_attack_prediction_db**
   - URL: [database-1.cfwmkaw8o6bp.us-east-1.rds.amazonaws.com]
   - Description: Connection to the heart attack data base in amazon RDS postgres    SQL.


### Data Analysis and Visualization

- **visuals_trial.ipynb** and **more_Visualization.ipynb** These files present our response to the research questions utilizing visualizations created with the Folium library and Plotly. Through these visualizations, we offer insights and analysis to address the research question effectively.


### Data Analysis 

- While the majority of the data collected aligns with expectations and demonstrates predictable patterns, certain unanticipated results have emerged due to key influence. Given that the data compilation spanned globally and accounted for numerous variable, it is conceivable that the sample size of 8,763 patients may not have been sufficiently large to encapsulate a definitive accuracy in the datas representation.

- The analysis reveals that adequate sleep and medication compliance are linked to a lower incidence of heart attacks. Moreover, it’s observed that people adhering to exceptionally healthy diets often register a higher risk of heart attacks. This counterintuitive result could be related to a phenomenon known as the “health-conscious worker effect,” where individuals who are proactive about their health are more likely to get regular check-ups. Such vigilance may lead to a higher reported incidence of heart issues simply because their conditions are more likely to be diagnosed then those less health-conscious


#### Model Training with Original Data

Random Forest Classifier, K-Nearest Neighbors (KNN), and Logistic Regression models are trained using the original dataset without any resampling techniques. The steps involved are as follows:

1. **Data Preparation**: The dataset is preprocessed to handle missing values, encode categorical variables, and scale numerical features if necessary. This ensures that the data is in a suitable format for training the models.

2. **Feature Selection**: Relevant features are selected based on their importance using techniques such as feature importance scores, recursive feature elimination, or domain knowledge. This helps improve the models' performance by focusing on the most informative features.

3. **Model Training**: Each model (Random Forest Classifier, KNN, and Logistic Regression) is instantiated and trained using the preprocessed dataset. During training, the models learn patterns and relationships between the input features and the target variable (heart attack risk).

4. **Model Evaluation**: After training, the performance of each model is evaluated using various metrics such as accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC). These metrics provide insights into how well the models are able to classify instances of heart attack risk.

#### Model Training with Resampled Data

The impact of different resampling techniques on model performance is explored for each model (Random Forest Classifier, KNN, and Logistic Regression). Resampling techniques are used to address class imbalance in the dataset, where one class (low-risk of heart attack) is significantly more prevalent than the other class (high-risk of heart attack). 

##### Data Splitting (training, testing and new testing set)
The data was split into training and testing sets using the `train_test_split` function from scikit-learn. Further splitting was performed on the testing set to create a new testing set using the same function. This ensured that the model was trained on one set of data and tested on another independent set, reducing the risk of overfitting and providing a more reliable evaluation of its performance.


##### The resampling techniques:


1. **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic samples for the minority class to balance the class distribution.

2. **ADASYN (Adaptive Synthetic Sampling)**: Similar to SMOTE, but adjusts the synthetic samples based on the local density of minority class instances.

3. **RandomOverSampler**: Randomly samples the minority class with replacement to match the majority class's size.

4. **RandomUnderSampler**: Randomly removes samples from the majority class to match the minority class's size.

5. **ClusterCentroids**: Under-samples the majority class by clustering the data points and keeping centroids of the clusters.

For each resampling technique, the dataset is resampled, and each model (Random Forest Classifier, KNN, and Logistic Regression) is trained using the resampled data. The trained models are then evaluated using the same evaluation metrics as the models trained with the original data. This allows for a comparison of how different resampling techniques affect each model's ability to predict heart attack risk accurately.







### Best Model: Data Resampling with RandomOverSampler, Grid Search, and RandomForestClassifier

This repository contains code for building and evaluating a predictive model for heart attack risk using resampling techniques with RandomOverSampler, grid search for hyperparameter tuning, and the RandomForestClassifier algorithm. The aim is to create a robust model that accurately predicts the risk of heart attack based on various features.

### Methodology

#### Data Resampling
- RandomOverSampler: To address class imbalance, the dataset is resampled using RandomOverSampler, which generates synthetic samples for the minority class to balance the class distribution.

#### Data Preprocessing
- Normalize Features: The feature matrix is normalized using MinMaxScaler after resampling to ensure consistent feature scaling.

#### Model Training
- RandomForestClassifier: The RandomForestClassifier algorithm is chosen for its ability to handle complex relationships in the data and its robustness to overfitting.

#### Hyperparameter Tuning
- Grid Search: Hyperparameters for the RandomForestClassifier are tuned using grid search with 5-fold cross-validation. The grid search explores combinations of hyperparameters such as the number of estimators, maximum depth, minimum samples split, and minimum samples leaf.

#### Results

The best hyperparameters found through grid search are:
- Max Depth: 30
- Min Samples Leaf: 4
- Min Samples Split: 10
- Number of Estimators: 300

The model achieves a testing data score of approximately 75% and a new testing data score of approximately 73%. Confusion matrices and classification reports are provided for both testing and new testing data, showing the precision, recall, and F1-score for each class.


### Setup

Ensure that you have the necessary Python libraries installed, including Pandas, NumPy, scikit-learn, psycopg2, and imbalanced-learn. Additionally, make sure you have access to a PostgreSQL database where the heart attack prediction dataset is stored.

### Instructions

1. Ensure you have access to the PostgreSQL database containing the heart attack prediction dataset.

2. Install the required Python libraries mentioned in the setup section.

3. Execute the code cells sequentially in your preferred Python environment (e.g., Jupyter Notebook, Google Colab Notebook, Jupyter Lab on AWS:3 (Amazon SageMaker)) to perform data retrieval, preprocessing, model training, and evaluation.

4. Review the results and analysis provided to understand the predictive performance of the models and make informed decisions regarding heart attack risk prediction.


### Data Sources

**Heart Attack Risk Prediction Dataset**
- URL[https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)

**Contries Now API**
- URL [https://countriesnow.space/api/v0.1/countries/capital] (https://countriesnow.space/api/v0.1/countries/capital)

