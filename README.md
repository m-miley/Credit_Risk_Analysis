# Credit_Risk_Analysis

- Resources
    - Python 3.7
    - jupyter notebook
    - Pandas, sklearn, imblearn

- Machine Learning Models
    - Logistic Regression
    - Balanced Random Forest Classifier
    - Easy Ensemble AdaBoost Classifier

- Sampling Techniques
    - Random Oversampling
    - SMOTE - synthetic minority oversampling technique
    - Cluster Centroids - undersampling
    - SMOTEENN - SMOTE + Edited Nearest Neighbor


## Overview

Data Analysis and Machine Learning project designed to predict credit risk for a money lending company.  Given that credit risk is largely unbalanced in classification, the implementation of resampling techniques to increase the effectiveness of the machine learning models is necessary.  Pre-processing techniques were applied to prepare the data for modeling such as encoding strings using .get_dummies() which splits the columns and maintains binary assignment, and StandardScaler to scale and standardize the data for balance and accuracy.  Our models were then initialized, trained (fit), predicted results, and tested for accuracy using accuracy_score generator, confusion matris, and further displayed in a classification report.

## Results

Each data set was split into train and test groups using train_test_split by sklearn.  The X features were then standardized to ensure that larger numbers weren't disproportionately impacting the models
![Screen Shot 2022-07-15 at 5 17 48 PM](https://user-images.githubusercontent.com/100544761/179318707-a73220b7-7543-4576-aa5c-afda4580bc01.png)

1. Model 1 involved oversampling the imbalanced data using RandomOverSampler.  A Logistic Regression model was then fit and used to predict outcomes.  Results were as follows:
![Screen Shot 2022-07-15 at 5 27 42 PM](https://user-images.githubusercontent.com/100544761/179319360-c10b6239-1f84-41c1-81df-e0066120f0ff.png)
