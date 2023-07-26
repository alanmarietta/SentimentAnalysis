# Importing necessary libraries
import pandas as pd
import nltk
import tensorflow_datasets as tfds  # <-- Added this
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
# Load the IMDb dataset from TensorFlow Datasets
(train_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)

# Convert TensorFlow dataset to lists
train_texts, train_labels = zip(*[(x.numpy().decode('utf-8'), y.numpy()) for x, y in train_data])
test_texts, test_labels = zip(*[(x.numpy().decode('utf-8'), y.numpy()) for x, y in test_data])

# Convert lists to pandas DataFrame
data = pd.DataFrame({
    'review': train_texts + test_texts,
    'sentiment': train_labels + test_labels
})

# Check the first few rows of the dataset to understand its structure
print(data.head())

# Initialize the stopwords and the WordNet lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocesses a given text:
    1. Converts to lowercase.
    2. Removes punctuation.
    3. Tokenizes and removes stopwords.
    4. Lemmatizes the words.
    5. Returns the preprocessed text as a string.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenization and stopwords removal
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing to the review column of the dataset
data['review'] = data['review'].apply(preprocess_text)

# Save the preprocessed data for further use
data.to_csv("data/preprocessed_reviews.csv", index=False)

print("Data preprocessing completed!")
