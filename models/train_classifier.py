import sys
import nltk
import pandas as pd
import sqlite3
import re
import time
import numpy as np
import warnings
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score,recall_score, make_scorer, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import hmean
warnings.simplefilter('ignore')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    path = 'sqlite:///'+database_filepath
    engine = create_engine(path)
    df = pd.read_sql('SELECT * FROM message',engine)

    X = df['message']
    Y = df.iloc[:,4:40]
    return X, Y, Y.columns

def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=50)))
        ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    TN = []
    FP = []
    FN = []
    for i in range(y_pred.shape[1]):
        TN.append(confusion_matrix(np.array(Y_test)[:,i],y_pred[:,i])[1,1])
        FP.append(confusion_matrix(np.array(Y_test)[:,i],y_pred[:,i])[1,0])
        FN.append(confusion_matrix(np.array(Y_test)[:,i],y_pred[:,i])[0,1])
    precision = np.sum(TN) / (np.sum(TN) + np.sum(FN))
    recall = np.sum(TN) / (np.sum(TN) + np.sum(FP))
    print('Model F1_micro_averaged is {}'.format(hmean([precision,recall])))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()