# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from feature_engineering import word2vec_features
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download
nltk.download('punkt')

# Load the preprocessed data
data = pd.read_csv("data/preprocessed_reviews.csv")

# Splitting the data
texts = data['review'].values
labels = data['sentiment'].values
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2)

# Train Word2Vec model on the training data and then get the embeddings
texts_train_tokenized = [word_tokenize(text) for text in texts_train]
model = Word2Vec(sentences=texts_train_tokenized, vector_size=100, window=5, min_count=1, workers=4)
model.train(texts_train_tokenized, total_examples=len(texts_train_tokenized), epochs=10)

X_train = word2vec_features(texts_train, pretrained_model=model)
X_test = word2vec_features(texts_test, pretrained_model=model)

# Scale the embeddings
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000)  # Increasing max_iter to ensure convergence
model.fit(X_train, labels_train)

# Model Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(labels_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
