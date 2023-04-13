# 4400_FlightReview_NLP

## Title: Air Travel Satisfaction Prediction With Natural Language Processing Project 


## Abstract:

## Documentation:

### Introduction  
**What is the problem?**   
In this project, we aim to develop a predictive model to determine whether airline passengers will recommend their trip based on the quality of service they received during their air travel. The motivation behind this project is to help leading airline companies understand the priorities of their passengers, enabling them to optimize their resources and efforts in improving service quality. Additionally, this model will provide insights into the cognitive processes of passengers as they decide whether to recommend the airline to others.

**Why it is interesting?**  
This problem is interesting because understanding passenger satisfaction and their likelihood to recommend the airline is crucial for companies to maintain their competitive advantage in the industry. By identifying the key factors that influence passengers' recommendations, airlines can enhance their service offerings and create a more enjoyable flying experience for their passengers. This, in turn, can lead to increased customer loyalty, positive word-of-mouth, and higher profitability for the airline companies. The use cases for this problem can be found in the strategic decision-making processes of airlines, crew training and development, and resource allocation for various in-flight services.

Our model seeks to answer several important questions: What aspects do senior passengers care about most? What expectations do loyal customers have during their air journeys? What are the bearable delay time boundaries for different passengers? And, which three features are the most important predictors of passengers' likelihood to recommend the airline in general? By addressing these questions, our project aims to provide valuable insights for airline companies to cater to the diverse needs and preferences of their passengers, ensuring a better flying experience for all. Ultimately, this study will contribute to the ongoing efforts of airlines to improve customer satisfaction, enhance brand loyalty, and maintain a strong competitive position in the market, as well as increase the likelihood of passengers recommending their services to others.

**What is the approach you propose to tackle the problem?**  


**Why is the approach a good approach compared with other competing methods?**  


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

For NLP model, we applied the 'customer_review' and 'recommended' feature. To run the NLP model, you will first need to install torch related libraries on your local computer:  
```
pip uninstall torch torchtext -y

pip install torch==1.9.0 torchtext==0.10.0

pip install spacy

!python -m spacy download en_core_web_sm

!pip install --upgrade torchtext

```
And also import the following libraries:  
```
import pandas as pd
from textblob import TextBlob

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.datasets import IMDB
from sklearn.model_selection import train_test_split

import spacy
```

To build an NLP model for classifying customer reviews as positive or negative, we'll first preprocess the text data by tokenizing, removing stop words, and stemming or lemmatizing the words. Then, we'll represent the text using a numerical format such as word embeddings or bag-of-words matrix.

Next, we'll use a Recurrent Neural Network (RNN) with a Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) architecture to model the sequence of words in each review. The RNN will take in the numerical representation of the text and output a probability distribution over the three classes. We'll train the model on a portion of the dataset, evaluate its performance on a validation set, and tune its hyperparameters to optimize performance.

In summary, our goal is to build an NLP model that can accurately classify customer reviews as positive, negative, or neutral. We'll achieve this by using an RNN with an LSTM or GRU architecture to model the sequence of words in each review, and training the model using a numerical representation of the text.

## Results

**Main results**

**Natural Language Process Model**

The model has been trained for 3 epochs, and we can observe that the training accuracy does not improve significantly over these epochs. After the first epoch, the training accuracy is 66.23%, which increases slightly to 66.64% after the second and third epochs. The validation accuracy after the third epoch is 66.81%. These results indicate that the model might not be learning effectively from the data, as the accuracy levels are not very high and do not show significant improvement. The validation accuracy is arround 66.81%.


**Supplementary results**

**Natural Language Process Model**

The model architecture chosen for this task is a bidirectional LSTM with the following parameters:

Input dimension: Length of the TEXT vocabulary (INPUT_DIM)
Embedding dimension: 100 (EMBEDDING_DIM)
Hidden dimension: 128 (HIDDEN_DIM)
Output dimension: 1 (OUTPUT_DIM)
Number of layers: 2 (N_LAYERS)
Bidirectional: True (BIDIRECTIONAL)
Dropout: 0.5 (DROPOUT)
Padding index: Index of the padding token in TEXT vocabulary (PAD_IDX)
The optimizer chosen for training is Adam with a default learning rate, and the loss function is BCEWithLogitsLoss. The model is trained for 3 epochs, and the batch size is set to 64.

The parameter choices for the model architecture and training process seem reasonable, but the results indicate that the model might not be learning effectively from the data. This might be due to various factors such as insufficient training data, suboptimal hyperparameter choices, or the model architecture not being suitable for the task. Further experiments with different model architectures, hyperparameters, and additional training data might help in improving the performance.

## Discussion

**Natural Language Process Model**

The results obtained from the experiments indicate that the model's performance is not as good as we had hoped for. There is no significant improvement in accuracy across the epochs. This suggests that the model might not be effective in learning from the data. Here are some potential issues and possible solutions we plan to implement:

Insufficient training data: In order to increase the speed in trying we limit to 500 data sample might be too small for the model to learn effectively. A larger dataset could help the model learn more complex patterns and improve its performance. We will Increase the size of the dataset by including more records or by using data augmentation techniques.

Suboptimal hyperparameter choices: The current hyperparameter choices might not be the best fit for this problem. The learning rate, number of layers, hidden dimensions, dropout rate, and other hyperparameters could be tuned to improve the model's performance. We would Perform a grid search or use Bayesian optimization to find the optimal hyperparameters for this problem.

Preprocessing and tokenization: The quality of the input data and its preprocessing can have a significant impact on the model's performance. We will investigate the data preprocessing steps and improve them if necessary. This may include better tokenization, removing irrelevant information, or using a more suitable tokenizer for the task.