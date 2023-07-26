import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

def tfidf_features(texts, max_features=1000):
    """
    Transform a list of texts into a TF-IDF feature matrix.
    
    Args:
    - texts (list of str): The input texts.
    - max_features (int): The maximum number of features (unique words).

    Returns:
    - numpy.ndarray: The TF-IDF feature matrix.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    feature_matrix = vectorizer.fit_transform(texts)
    return feature_matrix.toarray()

def word2vec_features(texts, embedding_size=100, pretrained_model=None):
    """
    Transform a list of texts into a matrix of word2vec embeddings.
    Each text is represented as the average of its word embeddings.

    Args:
    - texts (list of str): The input texts.
    - embedding_size (int): Size of the word embeddings.

    Returns:
    - numpy.ndarray: The matrix of averaged word embeddings.
    """
    # Tokenize texts
    tokenized_texts = [word_tokenize(text) for text in texts]
    
    if pretrained_model is None:
        # Train a Word2Vec model if none is provided
        model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_size, window=5, min_count=1, workers=4)
        model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=10)
    else:
        model = pretrained_model
    
    embeddings = []
    for tokens in tokenized_texts:
        avg_embedding = np.mean([model.wv[token] for token in tokens if token in model.wv.index_to_key], axis=0)
        embeddings.append(avg_embedding)

    return np.array(embeddings)
