# -*- coding: utf-8 -*-
"""
hate_speech_offensive_language_eda.ipynb

This script uses machine learning models to classify a given text as hate speech, offensive speech or neither.

"""

import pandas as pd
import numpy as np
import re
import seaborn as sns
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# Download the Dataset (replace with your own download method)
#!kaggle datasets download -d thedevastator/hate-speech-and-offensive-language-detection

# Load the Dataset (replace with your own file path)
data = pd.read_csv('train.csv')

# Data Preprocessing

def remove_special_char(text):
  return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_urls(text):
  return re.sub(r'http\S+', '', text)

def remove_usernames_hashtags(text):
  return re.sub(r'@\w+|#\w+', '', text)

def remove_extra_spaces(text):
  return re.sub(r'\s+', ' ', text)

def clean_text(text):
  text = text.lower()
  text = remove_special_char(text)
  text = remove_urls(text)
  text = remove_usernames_hashtags(text)
  text = remove_extra_spaces(text)
  return text

data['cleaned_tweet'] = data['tweet'].apply(clean_text)

# Feature Engineering (TF-IDF)

from collections import Counter

words = data['cleaned_tweet'].apply(lambda x: [word for word in x.split()])
word_count = Counter([word for sublist in words for word in sublist])

tfidf = TfidfVectorizer(max_features=5000)
tweet = list(data['cleaned_tweet'])
X_train, X_test, y_train, y_test = train_test_split(tweet, data['class'], test_size=0.2, random_state=42)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#  Model Training and Evaluation

# Logistic Regression

model_lr = LogisticRegression()
model_lr.fit(X_train_tfidf, y_train)
y_pred_lr = model_lr.predict(X_test_tfidf)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
with open("model/model_lr.pkl", "wb") as f:
  pickle.dump(model_lr, f)

# Random Forest

model_rf = RandomForestClassifier()
model_rf.fit(X_train_tfidf, y_train)
y_pred_rf = model_rf.predict(X_test_tfidf)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
with open("model/model_rf.pkl", "wb") as f:
  pickle.dump(model_rf, f)

# Decision Tree

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_tfidf, y_train)
y_pred_dt = model_dt.predict(X_test_tfidf)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
with open("model/model_dt.pkl", "wb") as f:
  pickle.dump(model_dt, f)

# Confusion Matrix Example (using Seaborn)
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Assuming you have predicted labels (y_pred) for each model
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Repeat for Random Forest and Decision Tree predictions
# ... (your existing code till the TF-IDF vectorizer)

# User Input
user_input = input("Enter your text: ")

# Preprocess the user input
cleaned_user_input = clean_text(user_input)

# Transform the user input using the fitted TF-IDF vectorizer
user_input_tfidf = tfidf.transform([cleaned_user_input])

# Load the trained models (assuming they are saved in 'model' directory)
model_lr = pickle.load(open('model/model_lr.pkl', 'rb'))
model_rf = pickle.load(open('model/model_rf.pkl', 'rb'))
model_dt = pickle.load(open('model/model_dt.pkl', 'rb'))

# Make predictions using each model
prediction_lr = model_lr.predict(user_input_tfidf)[0]
prediction_rf = model_rf.predict(user_input_tfidf)[0]
prediction_dt = model_dt.predict(user_input_tfidf)[0]

# Print the predictions
print("Logistic Regression Prediction:", prediction_lr)
print("Random Forest Prediction:", prediction_rf)
print("Decision Tree Prediction:", prediction_dt)
