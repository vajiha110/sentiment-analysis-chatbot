# sentiment-analysis-chatbot
pip install flask

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample 
import nltk
from nltk.corpus import stopwords
import string
import joblib
from flask import Flask, request, render_template

# Download NLTK data
nltk.download('stopwords')

# Load the dataset
train_df = pd.read_csv('C:/Users/Sea Farer/Downloads/Documents/flask_app/app.py/train1.csv')

train_df

print(train_df['sentiment'].value_counts())

test_df = pd.read_csv('C:/Users/Sea Farer/Downloads/Documents/flask_app/app.py/test.csv', encoding='latin1')

test_df

# Handle missing values in 'selected_text' column
train_df['selected_text'] = train_df['selected_text'].fillna('')
test_df['selected_text'] = test_df['text'].fillna('')

# Handle missing values in 'text' column
train_df['text'] = train_df['text'].fillna('')
test_df['text'] = test_df['text'].fillna('')

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    # Keep words with minimum length to avoid removing useful information
    tokens = [word for word in tokens if len(word) > 2 and word not in stopwords.words('english')]
    return ' '.join(tokens)


print(train_df[['text', 'selected_text']].head(10))


train_df['processed_text'] = train_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# Step 1: Downsample the neutral class to handle imbalance
neutral_df = train_df[train_df['sentiment'] == 'neutral']
positive_df = train_df[train_df['sentiment'] == 'positive']
negative_df = train_df[train_df['sentiment'] == 'negative']

neutral_downsampled = resample(
    neutral_df, replace=False, n_samples=len(negative_df), random_state=42
)


# Combine balanced data
balanced_df = pd.concat([neutral_downsampled, positive_df, negative_df])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare the data for model training
X = balanced_df['processed_text']
y = balanced_df['sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})

# Step 2: Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(y_train.value_counts(), y_test.value_counts())


model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
    MultinomialNB(alpha=0.1)
)

model.fit(X_train, y_train)


# Step 4: Evaluate the model
y_pred = model.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
print(report)


# Step 5: Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                              display_labels=['Negative', 'Neutral', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


sample = ["I love this product!", "This is the worst experience."]
print(model.predict(sample))


# Save the model
joblib.dump(model, 'sentiment_model.pkl')











