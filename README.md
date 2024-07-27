# Car-Insurance-Claim-Probability-Prediction
PREDICTIVE MODEL – CAR INSURANCE CLAIM PROBABILITY (BASED ON COMPREHENSIVE CAR POLICY FEATURES AND SAFETY RATINGS)

In this project, we are trying to predict the probability of claiming car insurance by a customer of a car insurance company, based on historical trends in insurance claimed by customers and information about insured customers & their car features.

Objective - 
To understand the factors that influence claim frequency & severity in six months & enable insurance companies to assess risk better & determine appropriate premiums for policyholders.

Libraries used - 
Numpy, Pandas, Matplotlib, Column Transformer, StandardScaler, OneHotEncoder, Classification Report, Confusion Matrix, accuracy score & Roc-auc-score

Models Considered - 
Logistic Regression, Support Vector Classifier, Gaussian Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, Ada Boosting

Our data set contains 43 columns including categorical & numerical columns. Divided dataset into independent features & dependent features.
We separated independent features into numerical (15) & categorical columns (27) for pre-processing them individually. 

In numerical columns,
- We detected outliers using "Box-plot" and replaced them using the "KNN Imputer" technique.
- Using "Pearson's correlation" analysis, we observed that input columns had no multicollinearity.

In categorical columns,
- We performed the "Chi-square test" to check the significant relationship of the categorical columns with the dependent feature, based on the p-value.
- Out of 27 categorical columns, only 11 columns were found to have a significant relationship, & 16 did not have, considering p-values.

Using the "Column Transformer" method, we transformed all independent features simultaneously by
- feature scaling (Standardization) on numerical columns using the "Standard Scaler" method, &
- encoding categorical columns using the "OneHotEncoder" method.

We fixed the class imbalance issue in our dependent feature using the "SMOTE technique".

After testing different models, the Random Forest Classifier gave the best "roc-auc-score" of 60% & we performed feature extraction using this as our base model.
