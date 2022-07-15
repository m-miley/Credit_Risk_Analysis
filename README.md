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

1. **Model 1**  -  RandomOverSampler + LogisticRegression:
![Screen Shot 2022-07-15 at 5 27 42 PM](https://user-images.githubusercontent.com/100544761/179319360-c10b6239-1f84-41c1-81df-e0066120f0ff.png)
Offers a decent predictability for low_risk loans.  A near perfect *precision* score as most low_risk predictions were correct, however, the *recall* or *sensitivity* score indicates that of the actual low_risk loans, a lesser 86 percent will be predicted to be low_risk, which is decent.  Notice how the high_risk *precision* rating is virtually zero, meaning that 3 percent of predicted high_risk loans were actually high_risk.  On the other hand, with a *sensitivity* rating of 72 percent, a good number of all the actual high_risk loans were successfully predicted.  So, in other words, if *precision* is important, this model isn't desirable.  If *recall* or *sensitivity* is important, meaning if we want to predict as many high risk loans as possible (which I suggest for the purposes of this analysis), this model is moderately capable of proper classification.
