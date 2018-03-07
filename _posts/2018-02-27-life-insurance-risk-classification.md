---
tags:
  - python
  - classification
  - resampling
---

Predicting risk classification for life insurance applicants

{% include toc %}

## Background
With the proliferation of the internet, many transactions take place online and are not limited to online shopping, banking transactions, booking of air tickets. Consumers are now used to the speed of these transactions and this is the primary reason to improve on the life insurance application process.

The current life insurance application process takes an average of 30 days where customers provide extensive information to identify risk classification and eligibility, and may also include medical examinations. The development of a predictive model that classifies risk would expedite and make the insurance application process less labour intensive and in turn attract more customers.


## Goals
This is my capstone project for the Data Science Immersive Course that I am taking in General Assembly. Hence, I would be exploring the use of various machine learning techniques in order to build a predictive model that can accurately classify risk for life insurance applicants.


## Data
The dataset used for this project is the Prudential Life Insurance Dataset obtained from [Kaggle] (https://www.kaggle.com/c/prudential-life-insurance-assessment){:target="_blank"}. The data consists of 59381 rows (applications) with 128 variables describing attributes of life insurance applicants. Besides the target and id variables, there are 13 continuous, 60 categorical, 5 discrete and 48 dummy variables. Other than the continuous variables such as BMI, Wt, Ht and Ins_Age, the remaining features are anonymized. 

Variables                     | Description
------------------------------|---------------
ID                            | A unique identifier associated with an application
Product Info 1-7              | A set of normalised variables relating to the product applied for
Ins_Age                       | Normalised age of applicant
Ht                            | Normalised height of applicant
Wt                            | Normalised weight of applicant
BMI                           | Normalised BMI of applicant
Employee Info 1-6             | A set of normalised variables related to the employment history of the applicant
InsuredInfo 1-6               | A set of normalised variables providing information about the applicant
Insurance History 1-9         | A set of normalised variables related to the insurance history of the applicant
Family Hist 1-5               | A set of normalised variables related to the famikly history of the applicant
Medical History 1-41          | A set of normalised variables related to the medical history of the applicant
Medical Keyword 1-48          | Dummy variables relating to the presence of a medical keyword associated with the application
Response                      | Target ordinal variable relating to the final decision associated with an application


## Evaluation Metric
The metric used for evaluation is the Quadratic Weighted Kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0.

The response variable has 8 possible ratings from 1 to 8. Each application is characterized by a tuple [![eaeb]({{ site.url }}{{ site.baseurl }}/images/capstone/eaeb.jpg)]({{ site.url }}{{ site.baseurl }}/images/capstone/eaeb.jpg), which corresponds to its scores by Rater A (actual risk) and Rater B (predicted risk).  The quadratic weighted kappa is calculated as follows.

First, an N x N histogram matrix O is constructed, such that ![![oij]({{ site.url }}{{ site.baseurl }}/images/capstone/oij.jpg)]({{ site.url }}{{ site.baseurl }}/images/capstone/oij.jpg) corresponds to the number of applications that received a rating i by A and a rating j by B. An N-by-N matrix of weights, w, is calculated based on the difference between raters' scores:

[![wij]({{ site.url }}{{ site.baseurl }}/images/capstone/wij.jpg)]({{ site.url }}{{ site.baseurl }}/images/capstone/wij.jpg)

An N-by-N histogram matrix of expected ratings, E, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that E and O have the same sum.

From these matrices, the quadratic weighted kappa is calculated as: 

[![k]({{ site.url }}{{ site.baseurl }}/images/capstone/k.jpg)]({{ site.url }}{{ site.baseurl }}/images/capstone/k.jpg)

The Cohen Kappa Scoring scale below is referenced from http://www.statisticshowto.com/cohens-kappa-statistic/.

Kappa Score | Scale
------------|--------------------------
<0          | agreement equivalent to chance
0.1 – 0.20  | slight agreement
0.21 – 0.40 | fair agreement
0.41 – 0.60 | moderate agreement
0.61 – 0.80 | substantial agreement
0.81 – 0.99 | near perfect agreement
1           | perfect agreement




## Exploratory Data Analysis
**Target Variable**

From the countplot below, the target variable 'response' is imbalanced with 8 as the majority class represented in 33% of the records. The minority classes, 3 and 4 only make up 2% of the records respectively.

[![target]({{ site.url }}{{ site.baseurl }}/images/capstone/target.png)]({{ site.url }}{{ site.baseurl }}/images/capstone/target.png)


**Feature Engineering**

As the data has been deliberately anonymized, it is a challenge to engineer new features as we do not know what each feature represents other than BMI, age, height and weight. Logically, the features added below should improve the predictions.


* **Interaction term between BMI and Age** - Intuitively, a high BMI coupled with old age would typically result in a worse risk classification.


* **Count of medical keywords** - It is likely that more medical keywords would result in a worse classification.


* **Count of null features per application** - With more null features, it gets more difficult to assess an application and may result in a worse classification.


* **Generation of two new features from Product_Info_2** - Product_Info_2 consist of both a character and number component. The relationship between the target and both components should also be considered.

[![pi2]({{ site.url }}{{ site.baseurl }}/images/capstone/pi2.png)]({{ site.url }}{{ site.baseurl }}/images/capstone/pi2.png)




**Missing Data**

*  The 8 variables medical_history_10, medical_history_32, medical_history_24, medical_history_15, family_hist_5, family_hist_3, family_hist_2 and insurance_history_5 have more than 35% of missing data and would be dropped.

*  The remaining missing data is imputed with the mean of the respective features.

Features            | Number of Missing Values |Percentage of Missing Values/%
--------------------|--------------------------|-----------------------------
medical_history_10  |   58824                  |99.1
medical_history_32  |   58274                  |98.1
medical_history_24  |   55580                  |93.6
medical_history_15  |   44596                  |75.1
family_hist_5       |   41811                  |70.4
family_hist_3       |   34241                  |57.7
family_hist_2       |   28656                  |48.3
insurance_history_5 |   25396                  |42.8
family_hist_4       |   19184                  |32.3
employment_info_6   |   10854                  |18.3
medical_history_1   |    8889                  |15.0
employment_info_4   |    6779                  |11.4
employment_info_1   |      19                  |0.03


## Feature Selection 

**Selection of Continuous Feature**

*  The heatmap for continuous variables is created such that features that exhibit multicollinearity are dropped. There is high correlation between bmi and wt at 0.85 and bmi-age and ins-age at 0.88. Hence, wt and ins_age are dropped.

[![heatmap]({{ site.url }}{{ site.baseurl }}/images/capstone/heatmap.png)]({{ site.url }}{{ site.baseurl }}/images/capstone/heatmap.png)

**One Hot Encoding of Categorical Features**
*  Categorical variables need to be processed through one-hot encoding before machine learning can take place. Due to the large number of categorical features and their unique inputs, there is a total of 1010 features after one hot encoding.

**Removing features with near zero variance**
*  The features with near zero variance are removed as they would not contribute towards predicting the target variable. Upon removing those features, there is a total of 91 features remaining in the dataframe.

**Variable Selection using SelectKBest and Chi2**
*  A chi-square test is performed to select the best 30 categorical features for machine learning.

Upon completion of the steps above, the following 40 features are selected:
product_info_4, ht, bmi, bmi_age, employment_info_1, employment_info_4, employment_info_6, family_hist_4, med_col, countna, medical_keyword_3, medical_keyword_15, medical_keyword_23, medical_keyword_25, medical_keyword_37, medical_keyword_40, medical_keyword_48, product_info_2_3, product_info_2_4, product_info_2_5, product_info_2_7, product_info_3_10, employment_info_3_3, employment_info_5_3, insuredinfo_1_2, insuredinfo_6_2, family_hist_1_2, medical_history_1_7.9621722253, medical_history_2_112, medical_history_2_162, medical_history_4_2, medical_history_12_3, medical_history_16_3, medical_history_18_2, medical_history_23_3, medical_history_28_2, product_info_2_char_1, product_info_2_num_1, product_info_2_num_2, product_info_2_num_4


## Interpretation of Radviz Plot 

The Radviz plot projects an N-dimensional data set into simple 2D space where the influence of each dimension can be interpreted as a balance between the influence of all dimensions.

The Radviz plot shows the data points are clustered together and hence predicting risk classification would be challenging. 

[![radviz]({{ site.url }}{{ site.baseurl }}/images/capstone/radviz.png)]({{ site.url }}{{ site.baseurl }}/images/capstone/radviz.png)



## Resampling Data

The data is split into training and test sets with continuous features scaled.

```python
ss = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
Xs_train, Xs_test = ss.fit_transform(X_train), ss.fit(X_train).transform(X_test)
```


A pipeline is used to resample the training data to 8000 records for each target class using Synthetic Minority Over-sampling Technique (SMOTE) and RandomUnderSampler concurrently. For SMOTE, the minority class is over-sampled by taking each minority class sample and introducing synthetic data points along the line segments joining the k minority class nearest neighbors

```python
pipeline = make_pipeline(SMOTE(ratio={1:8000,2:8000,3:8000,4:8000,5 :8000, 6:8000, 7:8000}, random_state=1), RandomUnderSampler(ratio={8:8000}, random_state=1))

X_train_res, y_train_res = pipeline.fit_sample(Xs_train, y_train)
```

## Machine Learning - Classification

The various classification models below are developed and the low Quadratic Weighted Kappa Scores confirmed that classification algorithms are not a good approach for this problem.  

Model                    |Quadratic Weighted Kappa Score
-------------------------|-------------------------------
Logistic Regression      | 0.444
Support Vector Classifier| 0.425
K Neighbors Classifier   | 0.309
Decision Tree Classifier | 0.285
Extra Trees Classifier   | 0.441
Random Forest Classifier | 0.430
Bagging Classifier       | 0.438
Adaboost Classifier      | 0.422
Voting Classifier        | 0.423




## Machine Learning - Regression

An xgboost (eXtreme Gradient Boosting) regression model is developed with the parameters below.

Parameters             |   Values 
-----------------------|-----------------
max_depth              |     7 
eta                    |   0.05
silent                 |     1
min_child_weight       |    360
subsample              |   0.85
early_stopping_rounds  |    10
objective              |  reg:linear
eval_metric            |   rmse
colsample_bytree       |    0.3
num_rounds             |    720  

The predicted target values rounded off to the nearest number, yielded a test score of 0.512 which is higher than any of the classification models. In order to improve on the Quadratic Weighted Kappa score, the offsets from the train predictions are optimised using the fmin_powell function and applied it to the test predictions.

Model                         |  Quadratic Weighted Kappa Score
------------------------------|--------------------
xgboost                       | 0.512
xgboost with optimised offsets| 0.575

The xgboost regression model with optimised offsets produced the best score of 0.575. 


## Conclusion
On hindsight, the quadratic weighted kappa metric imposes a heavier weight penalty in misclassifying a 1 as 8 compared to 2. Hence, classification models would not produce good results as they do not take the ordinal nature of the target into consideration. This would explain why a xgboost regression model with optimal offsets would produce better results.
[The python notebooks for this project can be found here](https://github.com/ryanchangky/projects/tree/master/capstone){:target="_blank"}.
For future work, I would be explore if a neural network model can produce better results.
