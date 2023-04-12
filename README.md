# 4400_FlightReview_NLP

## Title: 

## Abstract:

## Documentation 

### Introduction 

### Setup:

**Dataset Description**
Our dataset was gathered from the file capston_airlin_review3.csv, and we found this dataset on Kaggle. This original dataset includes airline reviews from 2006 to 20199 for popular airlines around the world with multiple choices and free test questions. This dataset contains 17 features, including 'airline', 'overall', 'author', 'review_date',' customer_review', 'aircraft', 'traveller_type', 'cabin', 'route', 'date_flown', 'seat_comfort', 'cabin_service', 'food_bev', 'entertainment', 'ground_service', 'value_for_money', 'recommended'. We added another field called "sentiment" to hold sentiment scores for each line of customer review. This flight customer review dataset contains 14211 rows of data after we dropped all the NaN values. Since this dataset contains a few categorical features and a rich text field as well, we decide to implement both Decision Tree Classifier model and Natural Language Processing model to make predictions and evaluate the prediction accuracy. 

Feature descriptions are provided as follows:  
airline: Name of the airline  
overall: Overall point given to the trip between 1 to 10  
author: Author of the trip  
review_date: Date of the Review  
customer_review: Review of the customers in free text format  
aircraft: Type of the aircraft  
traveller_type: Type of traveller (e.g. business, leisure)  
cabin: Cabin at the flight  
date_flown: Flight date  
seat_comfort: Rated between 1-5  
cabin_service: Rated between 1-5  
food_bev: Rated between 1-5  
entertainment: Rated between 1-5  
ground_service: Rated between 1-5  
value_for_money: Rated between 1-5  
recommended: Binary, target variable  


**Data scource**  
If you want to have a look at the original dataset, please find it here on Kaggle: https://www.kaggle.com/datasets/efehandanisman/skytrax-airline-reviews

To load dataset into python:

```
import pandas as pd

df_flight = pd.read_csv("capstone_airline_reviews3.csv", encoding="utf-8", encoding_errors='ignore').dropna()
```

**Decision Tree Classifier Model Setup**  
For Decision Tree Classier model, we use 'overall', 'seat_comfort', 'cabin_service', 'food_bev', 'entertainment', 'ground_service', 'value_for_money', 'sentiment' features. To train a decision tree classifier, we preprocess the dataset by encoding categorical variables into dummy variables, scaling numerical variables, and splitting the data into training and testing sets.

```
from sklearn.model_selection import train_test_split
dummy = pd.get_dummies(df_flight['airline'])
dummy_travel = pd.get_dummies(df_flight['traveller_type'])
dummy_cabin = pd.get_dummies(df_flight['cabin'])
df_flight = pd.concat([df_flight, dummy,dummy_travel,dummy_cabin], axis=1)
X = df_flight.drop(['author', 'review_date','route','recommended','aircraft','airline','customer_review','date_flown','traveller_type', 'cabin'], axis = 1)
y = df_flight['recommended']
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

```

We use scikit-learn's DecisionTreeClassifier with default hyperparameters, and evaluate the model on the testing set using accuracy, confusion matric, MSE and F1-score with the optimale min_sample_split value and optimal min_sample_leaf value. The aim is to predict customers response to recommend the airlines they took based on the satisfaction score of each service they enjoyed during flight. In addition, we  apply Bagging approach to understand the importnace of each feature contains in the training dataset.

```
from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier()
tree.fit(X_train, y_train)
```

Moreover, we use scikit-learn's RandomForestClassifier with default hyperparameters, and evaluate the model on the testing set using accuracy and MSE with the optimal max_depth and the optimal min_samples_split. 

```
from sklearn.ensemble import RandomForestClassifier
# Train the random forest
random_forest = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
random_forest.fit(X_train, y_train)
```

**Natural Language Process Model Setup**  
For NLP model, we applied the 'customer_review' and 'recommended' feature. 

To build an NLP model for classifying customer reviews as positive, negative, or neutral, we'll first preprocess the text data by tokenizing, removing stop words, and stemming or lemmatizing the words. Then, we'll represent the text using a numerical format such as word embeddings or bag-of-words matrix.

Next, we'll use a Recurrent Neural Network (RNN) with a Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) architecture to model the sequence of words in each review. The RNN will take in the numerical representation of the text and output a probability distribution over the three classes. We'll train the model on a portion of the dataset, evaluate its performance on a validation set, and tune its hyperparameters to optimize performance.

In summary, our goal is to build an NLP model that can accurately classify customer reviews as positive, negative, or neutral. We'll achieve this by using an RNN with an LSTM or GRU architecture to model the sequence of words in each review, and training the model using a numerical representation of the text.

