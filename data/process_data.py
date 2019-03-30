# import libraries
import sys
import langid
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def correct_language(X):
    '''
    Function that finds out the language of the tweet
    Input: X as DataFrame with text contained in "message" column
    Output: Dataframe X with additional language "lang" column
    '''
    lang = []
    message = X['message']
    
    for text in message:
        b = langid.classify(text)[0]
        lang.append(b)
        
    X['lang'] = lang
    return X

def load_data(messages_filepath, categories_filepath):
    '''
    Load both messages and categories
    Input: messages_filepath, categories_filepath - filepath and filenames of both needed csv files
    Output: DataFrame after merging "messages" with "categories"
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge two DataFrames together using "id" field
    df = messages.merge(categories, on = 'id', how = 'left')
    return df
    
def clean_data(df):
    '''
    Cleans the loaded dataframe
    Input: initial dataframe
    Output: cleaned initial dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # define names of each cathegory
    row = categories.iloc[0,:]
    category_colnames = row.str[:-2].tolist()
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 
    
    # replace "categories" columns in df with new category columns        
    df.drop('categories',axis = 1, inplace = True)
    df = pd.concat([df,categories], axis = 1)
    
    # clean the data
    # remove Nans
    df = df.drop_duplicates()
    
    # remove columns without any "1"
    df.drop('child_alone',axis = 1,inplace = True)
    
    # select only tweets written in english
    df = correct_language(df)
    df = df.loc[df['lang'] == 'en']
    df.drop('lang', axis = 1,inplace = True)
    df = df.loc[df['related'] != 2]
    return df
        
def save_data(df, database_filename):
    '''
    Saves transformed and cleaned data into a SQL Database
    Input: df - transformed and cleaned dataframe, database_filename - a filename for an SQL Db 
    '''
    path = 'sqlite:///'+str(database_filename)
    engine = create_engine(path)
    df.to_sql('message', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
