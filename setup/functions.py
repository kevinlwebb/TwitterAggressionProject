import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize(text):
    '''
    Returns list of clean words

    Parameters:
        text (str):The text to be cleaned.

    Returns:
        tokens (list):A list with clean words. 
    '''

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word.lower().strip() not in stop_words]

    return tokens