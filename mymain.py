import pickle
import os
import subprocess
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression

# import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

import time
from sklearn.metrics import roc_auc_score

system_specs = "Macbook Pro, 3.49 GHz, 24GB memory"
execution_times = []

# Data preprocessing
def preprocess_data(data):
    data['review'] = data['review'].str.replace('<.*?>', ' ', regex=True)
    return data

# Train a model with embeddings      C_value= 8.0, 10.0, 12.0
def train_model(X_train, y_train, C_value=10.0, solver='liblinear', use_embeddings=True):
    """
    Train a logistic regression model using provided training data.

    Parameters:
    - X_train: Training feature data (NumPy array or DataFrame).
    - y_train: Training labels.
    - C_value: Regularization strength for Logistic Regression.
    - solver: Solver to use in Logistic Regression.
    - use_embeddings: Ignored if X_train is already prepared.

    Returns:
    - model: Trained Logistic Regression model.
    """
    # Assume X_train is already a NumPy array; no further slicing required
    model = LogisticRegression(random_state=42, solver=solver, max_iter=2000, C=C_value)
    model.fit(X_train, y_train)
    return model

# Save predictions to a CSV file
def save_predictions(models, test_data):
    X_test_embeddings = test_data.iloc[:, -1536:].values
    probabilities = np.mean([model.predict_proba(X_test_embeddings)[:, 1]
                             for model in models], axis=0) if isinstance(models, list) else models.predict_proba(X_test_embeddings)[:, 1]
    submission = pd.DataFrame({'id': test_data['id'], 'prob': probabilities})
    submission.to_csv('mysubmission.csv', index=False)



#  Try using different hyper-parameter
# C_value = (8.0, 10.0, 12.0)  solver =(liblinear , saga, lbfgs  # test_size = 0.1, 0.2, 0.3

# auc_scores = []
start_time = time.time()

# Load and preprocess data
train_data = pd.read_csv(f'train.csv')
test_data = pd.read_csv(f'test.csv')
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Prepare training and test data
X_train = train_data.iloc[:, -1536:].values  # Converts DataFrame to NumPy array
y_train = train_data['sentiment']
X_test = test_data.iloc[:, -1536:]  # Keep X_test as a DataFrame

# Train model
model = train_model(X_train, y_train, C_value=10.0, solver='liblinear')
X_train = train_data.iloc[:, -1536:].values  # Converts DataFrame to NumPy array

execution_time = time.time() - start_time
execution_times.append(execution_time)

print(f"Execution Time - {execution_time:4f} seconds")

# Generate and save predictions on test data
save_predictions(model, test_data)

# Serialize the model
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# now need to train model using BERT and then map that to the format 
# that can then be used by the model we just trained in part 1

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import random
import re

from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

test = pd.read_csv("test.csv") 
train = pd.read_csv("train.csv")

X_test = test['review']
X_train = train.drop(columns=['sentiment'])
Y_train = train['sentiment']

# Perform some pre-processing.
# This is from https://campuswire.com/c/GB46E5679/feed/800


X_train['review'] = X_train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "their", "they", "his", "her", "she", "he", "a", "an", "and", "is", "was", "are", "were", "him", "himself", "has", "have", "it", "its", "the", "us"]

vectorizer = CountVectorizer(
    preprocessor=lambda x: x.lower(),  # Convert to lowercase
    stop_words=stop_words,             # Remove stop words
    ngram_range=(1, 4),               # Use 1- to 4-grams
    min_df=0.001,                        # Minimum term frequency
    max_df=0.5,                       # Maximum document frequency
    token_pattern=r"\b[\w+\|']+\b" # Use word tokenizer: See Ethan's comment below
)

dtm_train = vectorizer.fit_transform(X_train['review'])

print('finished creating dtm_train')

ones_train_indexes = Y_train[Y_train == 1].index.tolist()
zeros_train_indexes = Y_train[Y_train == 0].index.tolist()

ones_train = dtm_train[ones_train_indexes]
zeros_train = dtm_train[zeros_train_indexes]

# Perform the two sample t-test to reduce the vocabulary size
# This is from https://campuswire.com/c/GB46E5679/feed/801

# Compute the mean for each matrix along axis 0 (columns/features)
mean1 = np.array(ones_train.mean(axis=0)).flatten()
mean2 = np.array(zeros_train.mean(axis=0)).flatten()

# Compute the variance for each matrix along axis 0
var1 = np.array(ones_train.multiply(ones_train).mean(axis=0)).flatten() - (mean1 * mean1)
var2 = np.array(zeros_train.multiply(zeros_train).mean(axis=0)).flatten() - (mean2 * mean2)

# Number of samples in each matrix
n1 = ones_train.shape[0]
n2 = zeros_train.shape[0]

# Calculate the t-statistic and p-value for each feature
t_stat = (mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))
df = ((var1 / n1 + var2 / n2) * (var1 / n1 + var2 / n2)) / ((var1 * var1 / (n1 * n1 * (n1 - 1))) + (var2 * var2 / (n2 * n2 * (n2 - 1))))
p_values = 2 * stats.t.cdf(-1 * np.abs(t_stat), df)

sorted_indices = np.argsort(p_values)
n = 2000
top_indices = sorted_indices[:n]
top_p_values = p_values[top_indices]
dtm_train = dtm_train[:, top_indices]

vocab = vectorizer.get_feature_names_out()
vocab_words = list(vocab[top_indices])

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(dtm_train)


# Clean the test set reviews
test['review'] = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

# Transform the test data using the same vectorizer
dtm_test = vectorizer.transform(test['review'])
dtm_test = dtm_test[:, top_indices]

# Scale the test data using the same scaler
test_scaled = scaler.transform(dtm_test)

print('created test_scaled')

with open("test_scaled.pkl", "wb") as f:
    pickle.dump(test_scaled, f)

with open("vocab_words.pkl", "wb") as f:
    pickle.dump(vocab_words, f)

with open("dtm_test.pkl", "wb") as f:
    pickle.dump(dtm_test, f)


