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
    '''
    Loads data from SQL Database
    Input: filepath to an existing SQL Database
    Output: X-pandas Series with messages, Y - pandas dataframe containing categories,
            Y.columns - numpy array with category names
    '''
    
    path = 'sqlite:///'+database_filepath
    engine = create_engine(path)
    df = pd.read_sql('SELECT * FROM message',engine)

    X = df['message']
    Y = df.iloc[:,4:40]
    return X, Y, Y.columns

def tokenize(text):
    '''
    Function that transfers text to tokes
    Input: text as string, a unmodified plain text
    Output: tokens created from modified and cleaned text
    '''
    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def f1_micro_average(y_test,y_pred):
    '''
    Calculates s.c. micro average f1
    Input: y_test, y_pred - arrays with test and prediction values
    Output: micro average f1
    '''
    
    TN = []
    FP = []
    FN = []
    for i in range(y_pred.shape[1]):
        TN.append(confusion_matrix(np.array(y_test)[:,i],y_pred[:,i])[1,1])
        FP.append(confusion_matrix(np.array(y_test)[:,i],y_pred[:,i])[1,0])
        FN.append(confusion_matrix(np.array(y_test)[:,i],y_pred[:,i])[0,1])
    precision = np.sum(TN) / (np.sum(TN) + np.sum(FN))
    recall = np.sum(TN) / (np.sum(TN) + np.sum(FP))
    
    return hmean([precision,recall])


def build_model():
    '''
    Defines a ML Pipeline
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    
    parameters = {'vect__min_df': [1, 5],
              'clf__estimator__n_estimators': [50, 75],
             'clf__estimator__learning_rate':[0.1,1,10]}

    scorer = make_scorer(f1_micro_average)

    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10, cv = 2)

    return cv

    


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Calculates precision, recall and f1 metrics for each category and combines it into a single dataframe
    Input: y_test, y_pred - arrays with test and prediction values
    Output: res_metrics - a pandas dataframe with all metric values for each category
    '''
    y_pred = model.predict(X_test)
    
    res_metrics = []
    for i in range(y_pred.shape[1]):
        f1 = f1_score(Y_test.values[:,i],y_pred[:,i])
        precision = precision_score(Y_test.values[:,i],y_pred[:,i])
        recall = recall_score(Y_test.values[:,i],y_pred[:,i])
        res_metrics.append([f1,precision,recall])
        
    print('F1_micro_average: {}'.format(f1_micro_average(Y_test,y_pred)))
    print(pd.DataFrame(res_metrics, columns = ['f1','precision','recall'], index = category_names) )
    

def save_model(model, model_filepath):
    '''
    Saves resulting trained model into a pkl file
    '''
    
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
