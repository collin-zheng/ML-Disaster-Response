import sys

import time
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories .csv data and joins them into a Pandas dataframe.
    
    INPUTS:
        messages_filepath - file path of messages .csv data.
        categories_filepath - file path of categories .csv data.
    OUTPUTS:
        df - joined dataset as a Pandas dataframe.
    """
    
    # reading .csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # dataset join
    df = pd.merge(messages, categories, on = 'id')
    
    return df


def clean_data(df):
    """
    Clean the joined messages and categories dataset.
    The method first explodes the category column into 36 columns based on categories with a
    one-hot encoding of values.
    The method then cleans re-attaches the one hot encoded category table and finalises cleaning of the dataframe.
    
    INPUT:
        df - joined dataframe.
    OUTPUT:
        df - cleaned dataframe.
    """
    
    # Task 1: explode column column into one-hot encoded category columns
    # explode the category rows into 36 columns based on categories
    # i.e. related-1;request-0;offer-0... becomes related-1 | request-0 | offer-0...
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe for category name extraction
    # apply lambda function that removes unwanted final two characters for each current category name
    # i.e. 0 | 1 | 2... becomes related | request | offer...
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert df values to 1's and 0's by removing unwanted category names attached to each cell
    # e.g. related-1 to 1 and search_and_rescue-0 to 0.
    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = categories[col].astype(int)
        
    # cap all value 2 cells in 'related' column to 1
    categories = categories.clip(0,1)
    
    # Task 2: Clean up rest of df
    # drop original categories column from 'df'
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataframe into a SQLite .db database to a specified path. 
    
    INPUTS:
        df - cleaned dataframe.
        database_filename - name and path to store SQLite .db file. For example, DisasterMessages.db.
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index = False, if_exists = 'replace')


def main():
    """
    Main executable function.
    No changes made to Udacity template, except for a script execution timer.
    """
    
    start_time = time.time()    

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
        
    print("Script execution time: {} seconds |".format(round((time.time() - start_time), 1)))


if __name__ == '__main__':
    main()