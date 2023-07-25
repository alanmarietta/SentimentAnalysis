# Importing necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the IMDb dataset
# Note: You need to have the dataset in CSV format. Modify the path below accordingly.
data_path = "data/imdb_reviews.csv"
data = pd.read_csv(data_path)

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
