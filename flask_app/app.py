import sys
sys.path.append('..')  # Add the parent directory to the path

from flask import Flask, render_template, request
import torch
import torch.nn as nn  # Importing necessary PyTorch modules
from feature_engineering import word2vec_features
from preprocess import preprocess_text
from gensim.models import Word2Vec

app = Flask(__name__)

# Define your Logistic Regression Model architecture
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

# Initialize the model instance
input_dim = 100  # Make sure this matches with your training dimension
model = LogisticRegressionModel(input_dim)
w2v_model = Word2Vec.load("../models/word2vec_model.model")

# Load the saved state dictionary into the model
model.load_state_dict(torch.load('../models/movie_review_model.pth'))
model.eval()  # Set the model to evaluation mode

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form.get('review_text')
        
        # Preprocess the text and get features
        processed_text = preprocess_text(text)
        
        # Use the loaded Word2Vec model to get embeddings
        features = word2vec_features([processed_text], pretrained_model=w2v_model)

        # Make prediction
        output = model(torch.tensor(features).float())
        prediction = 'Positive' if output.item() > 0.5 else 'Negative'  # Note the use of .item() to get a scalar from the output tensor

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
