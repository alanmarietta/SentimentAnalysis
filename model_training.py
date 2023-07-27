import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from feature_engineering import word2vec_features
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import torch
import torch.nn as nn
import torch.optim as optim

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

# This part creates the Word2Vec model instance without immediately training it
w2v_model = Word2Vec(sentences=None, vector_size=100, window=5, min_count=1, workers=4)

# Build the vocabulary from your training data
w2v_model.build_vocab(texts_train_tokenized)

# Now you can train the model
w2v_model.train(texts_train_tokenized, total_examples=len(texts_train_tokenized), epochs=10)

X_train = word2vec_features(texts_train, pretrained_model=w2v_model)
X_test = word2vec_features(texts_test, pretrained_model=w2v_model)


# Scale the embeddings
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
labels_train_tensor = torch.FloatTensor(labels_train).view(-1, 1)  # Reshaping to column tensor
labels_test_tensor = torch.FloatTensor(labels_test).view(-1, 1)

# Logistic Regression using PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

input_dim = X_train_tensor.shape[1]
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the Model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, labels_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Model Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_outputs = (test_outputs > 0.5).float()
    accuracy = (test_outputs == labels_test_tensor).sum().item() / labels_test_tensor.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")

# Saving the model
model_path = "models/movie_review_model.pth"
torch.save(model.state_dict(), model_path)
# Saving w2v model to convert review text into embeddings
w2v_model.save("models/word2vec_model.model")
