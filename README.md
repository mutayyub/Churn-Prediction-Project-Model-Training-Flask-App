# Churn-Prediction-Project-Model-Training-Flask-App

Predicting whether a customer will **stay or leave (churn)** is a key challenge for businesses.  
In this project, I'll build a machine learning model to analyze customer data and predict churn.

---

## Dataset

* **Size:** 64,374 rows and 12 columns.
* **Target Variable:** `Churn` (1 = churned, 0 = stayed)
* **Key Features:** Usage Frequency, Total Spend, Last Interaction, etc.
* Removed **Payment Delay** column due to **data leakage** (causing unrealistic 100% accuracy).
* **Dataset**: [Link](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data)

## Exploratory Data Analysis (EDA)

We explored the dataset using the **four key pillars of EDA**:

1. **Composition**

   * Around *46.2%* of customers churned while the rest stayed.

2. **Distribution**

   * The data is spread out evenly across the board, no weird spikes or dips, just a nice, consistent spread.

3. **Comparison**

   * Churned customers generally had **lower usage frequency** and **lower spending** compared to retained customers.
   * Customers with **longer tenure** were less likely to churn.

4. **Correlation**

   * Overall, correlations were weak.
   * **Usage Frequency** and **Total Spend** show a slight *negative correlation* with churn.
   * **Age**, **Support Calls**, and **Tenure** had *positive correlations* with churn.

## Models Used

We trained and evaluated several machine learning models:

* Logistic Regression
* Support Vector Classifier
* KNeighbors Classifier
* Decision Tree, Random Forest
* Gradient Boosting, AdaBoost
* XGBoost     ---------->     ✅ (Best Model)

## Results

* After hyperparameter tuning, the **XGBoost Classifier** performed the best.
* **Accuracy:** 77.5% on the test set.
* Feature importance gain showed that Gender and Support calls were the strongest predictors of churn.

## Key Insights

* Customers with **high engagement and spending are less likely to churn**.
* The model can help businesses **identify at-risk customers** and **design targeted retention strategies**.

## 📌 Final Step

* Make a Flask app for real-world usage.

---

```

**⚡ Note on Dataset Usage**

This project originally came with a training dataset of ~450,000 rows and a separate test dataset of 64,374 rows. `To keep the workflow efficient and resource-friendly, I trained my model on the 64k dataset instead of the full 450k.`


**This choice was made to**:

- Reduce computation time and memory usage.

- Focus on demonstrating the end-to-end machine learning pipeline.

```
