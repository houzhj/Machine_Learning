
# Credit Card Fraud Detection using Gradient Boosting
- Introduction:
  - This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
  - The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
  - It contains only numerical input variables which are the result of a [PCA transformation](https://en.wikipedia.org/wiki/Principal_component_analysis).
  - Feature 'Class' is the target feature variable and it takes value 1 in case of fraud and 0 otherwise.
- Data [[Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)]
- [**Exploratory data analysis and oversampling**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Credit_Card_Fraud_Detection/credit_card_fraud_EDA_oversampling.ipynb)
- [**Hyperparameter Tuning**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/Credit_Card_Fraud_Detection/credit_card_fraud_hyperparameter.ipynb)

# Sentimential Analysis using IMDB Reviews (TFIDF + Gradient Boosting)
- Introduction:
  - IMDB dataset having 50,000 movie reviews for natural language processing or Text analytics.
  - The goal is to predict the "sentiment" class which takes value "positive" or "negative".
- Data [[Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)]
- [**Data treatment and feature engineering**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/imdb_data.ipynb)
  - [**More about creating features using TFIDF**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/tfidf.ipynb)
- [**Feature selection (Iterative Reduction, Boruta)**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/imdb_feature_selection.ipynb)
  - [**Random noise feature test**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/imdb_random_noise_test.ipynb): evaluate feature selection methods, according to whether they are able to identify random noise feautures.
  - [**Boruta analysis**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/imdb_boruta_analysis.ipynb): Conducting Burota feature selection using different combinations of algorithms (LGB,XGB, Random forest) and type of feature importance (gain,split,cover,weight,impurity).
- **Object-oriented programming (OOP) version**
  - [**Data treatment and feature selection**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/imdb_oop_data_feature.ipynb)
    - data cleaning
    - data engineering (creating TFIDF-based features)
    - Iterative reduction and Boruta feature selection approaches
  - [**Hyperparameter tuning**](https://github.com/houzhj/Machine_Learning/blob/main/ipynb/IMDB_Reviews/imdb_oop_hp.ipynb)
    - Random search
    - Visualization 
