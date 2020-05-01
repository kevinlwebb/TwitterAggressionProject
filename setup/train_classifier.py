import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load

from functions import tokenize


def load_data(database_filepath):
    '''
    Returns X (features) and Y (label)

    Parameters:
        database_filepath (str):The file path to the database.

    Returns:
        X (Array):A dataframe with the project data.  
        Y (Array):A dataframe with the project data.   
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('tweets', engine)
    X = df.cleaned_tweet.values
    Y = df.label.values
    return X, Y


def build_model():
    '''
    Returns a pipeline that incudes Bag of Words, TF-IDF, and Random Forest

    Returns:
        pipeline (Pipeline):A pipeline for the input data to follow before the classifier. 
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    return pipeline



def evaluate_model(model, X_test, y_test):
    '''
    Prints labels, the confusion matrix, and accuracy of the given model

    Parameters:
        model (Pipeline):A pipeline for the input data to follow before the classifier.
        X_test (Array):The array to be used for predictions.
        y_test (Array):The array to be used for comparisons.

    '''
   
    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    


def save_model(model, model_filepath):
    '''
    Saves model

    Parameters:
        model (Pipeline):A pipeline for the input data to follow before the classifier.
        model_filepath (str):The file path for the model to be saved.
        
    '''

    dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/TweetSentiment.db classifier.pkl')


if __name__ == '__main__':
    main()