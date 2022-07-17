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

Data Analysis and Machine Learning project designed to predict credit risk for a money lending company.  Given that credit risk is largely unbalanced in classification, the implementation of resampling techniques such as oversampling, undersampling, and combination sampling to increase the effectiveness of the machine learning models is necessary.  Pre-processing techniques were applied to prepare the data for modeling such as encoding strings using .get_dummies() which splits the columns and maintains binary assignment, and StandardScaler to scale and standardize the data for balance and accuracy.  Our models were then initialized, trained (fit), predicted results, and tested for accuracy using accuracy_score generator, confusion matris, and further displayed in a classification report.

## Results

Each data set was split into train and test groups using train_test_split by sklearn.  The X features were then standardized to ensure that larger numbers weren't disproportionately impacting the models
![Screen Shot 2022-07-15 at 5 17 48 PM](https://user-images.githubusercontent.com/100544761/179318707-a73220b7-7543-4576-aa5c-afda4580bc01.png)

1. **Model 1**  -  RandomOverSampler + LogisticRegression
![Screen Shot 2022-07-15 at 5 53 06 PM](https://user-images.githubusercontent.com/100544761/179321853-1e17afac-b6d3-437a-ab27-1894deedc406.png)
Random oversampling is when instances from the minority class are randomly selected and added to the training set until classes are balanced.  Model 1 offers a decent predictability for low_risk loans.  A near perfect *precision* score as most low_risk predictions were correct, however, the *recall* or *sensitivity* score indicates that of the actual low_risk loans, a lesser 86 percent will be predicted to be low_risk, which is decent.  Notice how the high_risk *precision* rating is virtually zero, meaning that 3 percent of predicted high_risk loans were actually high_risk.  On the other hand, with a *sensitivity* rating of 72 percent, a good number of all the actual high_risk loans were successfully predicted.  So, in other words, if *precision* is important, this model isn't desirable.  If *recall* or *sensitivity* is important, meaning if we want to predict as many actual high risk loans as possible (which I suggest for the purposes of this analysis), then this model is moderately capable of proper classification. 

2. **Model 2**  -  SMOTE Oversampling + Logistic Regression
![Screen Shot 2022-07-15 at 5 53 13 PM](https://user-images.githubusercontent.com/100544761/179321875-4df897e1-f38d-41fc-905f-178dcfba4670.png)
Model 2 implements a synthetic minority oversampling technique where new instances are interpolated from data in the minority class.  It views the nearest neighbors and creates new values.  The results of implementing SMOTE in this situation offers no significant improvement in predicting high_risk loans.  In fact, it predicts fewer actual high_risk loans which is a step in the wrong direction.  We know this as the "rec" (recall) score is lower by 2%.    

3. **Model 3**  -  ClusterCentroids Undersampling + LogisticRegression
![Screen Shot 2022-07-15 at 6 09 05 PM](https://user-images.githubusercontent.com/100544761/179322830-2942b20a-3c55-40f0-9004-6b58db8af227.png)
Cluster Centroid Undersampling is similar to SMOTE in that it generates synthetic data, however, it is representative of the majority and cuts down on the sample size from the majority class.  So far, the results from this model outperform other tests.  If we are trying to funnel as many high _risk loans into the proper category in prediction results, then this is our test.  The sensitivity, or recall, score is .78, the highest for our high_risk loans. Although, the low_risk sensitivity score dropped .10 points.  I think this is okay given the goal of our model.  

4. **Model 4**  -  SMOTEENN Combination Sampling+ LogisticRegression
![Screen Shot 2022-07-15 at 6 21 37 PM](https://user-images.githubusercontent.com/100544761/179323631-685bbd4b-442f-4f98-84cb-b57651c02f8e.png)
SMOTEENN is a combination sampling technique that oversamples the minority class using SMOTE, however, undersamples the majority class by removing nearest neighbors of data points belonging to two different classes.  The results show decreased recall score with high_risk loans.  Therefore, we should not use this model in production, compared to Model 3.

5. **Model 5**  -  Balanced Random Forest Classifier
![Screen Shot 2022-07-17 at 2 08 39 PM](https://user-images.githubusercontent.com/100544761/179421158-c156a109-79ac-401e-b9e2-0250cb450dd8.png)
The first of our Ensemble Learning machine learning models, Balanced Random Forest Classifier, randomly undersamples the imbalanced features data and applies the Random Forest Classifier, a combination of weaker, simpler, decision trees that provide a unified, stronger result.  Again, here, we see only marginal changes to our prediction results.  There is no significant positive difference that suggests this model is better than the others at predicting high_risk loan.  

6. **Model 6**  -  Easy Ensemble Classifier
![Screen Shot 2022-07-17 at 2 32 57 PM](https://user-images.githubusercontent.com/100544761/179421928-c523d975-ce1c-414f-a130-1e5c3e25f4a1.png)
The Easy Ensemble Classifier is an ensemble learning machine learning model comprised of AdaBoost learners.  It is an adaptive boosting technique where each iteration learns from the previous execution.  The next iteration is given a weight based on the outcomes and model is rerun until we've reach the specified limit, hopefully producing a robust outcome of prediction results.  We can clearly see here, that Model 6 produces the best results.  Scores in all significant categories are higher.  Precision scores and recall see a significant increase well into the 90th percentile.  Most importantly, of the 87 high_risk loans, 79 were predicted as high risk, resulting in a recall score of 91%.  This is a dramatic improvement from all other models in our analysis.  I recomment using this model for production if this is the extent of our research.

## Summary

In summary, I as mentioned above, I recommend using the Easy Ensemble Classifier machine learning model (EEC) if chosing between the models in our analysis.  It provides sufficient results in predicting actual high risk loans.  Though not terribly precise, it has shown to predict them with an sensitivity score of 91%.  This, presumably, was the goal of our analysis, to produce a machine learning model that predicts a high percentage of high risk laons.  Furthermore, the accuracy score was significantly higher in Model 6.  The accuracy score represents the number of correctly classified instances per total number of instances.  

Now, the next question would be "What percentage of predictability is acceptable?"  We will have to consult other business leaders and discuss as a team what we deem significant.

### Contact

Email: mrmileyy@gmail.com
[LinkedIn](https://www.linkedin.com/in/mileymarshall)