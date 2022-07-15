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

1. **Model 1**  -  RandomOverSampler + LogisticRegression
![Screen Shot 2022-07-15 at 5 53 06 PM](https://user-images.githubusercontent.com/100544761/179321853-1e17afac-b6d3-437a-ab27-1894deedc406.png)
Model 1 offers a decent predictability for low_risk loans.  A near perfect *precision* score as most low_risk predictions were correct, however, the *recall* or *sensitivity* score indicates that of the actual low_risk loans, a lesser 86 percent will be predicted to be low_risk, which is decent.  Notice how the high_risk *precision* rating is virtually zero, meaning that 3 percent of predicted high_risk loans were actually high_risk.  On the other hand, with a *sensitivity* rating of 72 percent, a good number of all the actual high_risk loans were successfully predicted.  So, in other words, if *precision* is important, this model isn't desirable.  If *recall* or *sensitivity* is important, meaning if we want to predict as many actual high risk loans as possible (which I suggest for the purposes of this analysis), then this model is moderately capable of proper classification. 

2. **Model 2**  -  SMOTE Oversampling + Logistic Regression
![Screen Shot 2022-07-15 at 5 53 13 PM](https://user-images.githubusercontent.com/100544761/179321875-4df897e1-f38d-41fc-905f-178dcfba4670.png)
Model 2 offers no significant improvement in predicting high_risk loans.  In fact, it predicts fewer actual high_risk loans which is a step in the wrong direction.  We know this as the "rec" (recall) score is lower by 2%.    

3. **Model 3**  -  ClusterCentroids + LogisticRegression
![Screen Shot 2022-07-15 at 6 09 05 PM](https://user-images.githubusercontent.com/100544761/179322830-2942b20a-3c55-40f0-9004-6b58db8af227.png)


4. **Model 4**  -  SMOTEENN + LogisticRegression
![Screen Shot 2022-07-15 at 6 21 37 PM](https://user-images.githubusercontent.com/100544761/179323631-685bbd4b-442f-4f98-84cb-b57651c02f8e.png)
