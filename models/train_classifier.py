import sys

import time

import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score

import pickle




def load_data(database_filepath):
    """
    Load .db file into a dataframe. Extract features and target variable information for modelling.
    
    INPUT:
        database_filepath - name and path of .db file containing the message and categories data.
    OUTPUTS:
        X - inputs/features (message data)
        Y - outputs/target (category of each message)
        cat_names - names of all 36 categories
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    cat_names = list(df.columns[4:])

    return X, Y, cat_names


def tokenize(text):
    """
    Tokenise messages for modelling.
    This includes:
        - tokenisation into word tokens,
        - lemmatisation (by noun & verb POS) with Nltk's WordNetLemmatizer()
        - stemming with Nltk's PorterStemmer()
    
    INPUT:
        text - message to be tokenised
    OUTPUT:
        text_tokenised - tokenised, lemmatised and stemmed messages
    
    """
    
     # replace all non-alphabetic and non-numerical characters with blank spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenise words
    tokens = word_tokenize(text)
    
    # instantiate lemmatiser object
    lemmatizer = WordNetLemmatizer()
    
    # instantiate stemmer object
    stemmer = PorterStemmer()
    
    text_tokenised = []
    for tok in tokens:
        # lemmtise token using noun as POS
        tok_temp = lemmatizer.lemmatize(tok)
        # lemmtise token using verb as POS
        tok_temp = lemmatizer.lemmatize(tok_temp, pos = 'v')
        # stem token
        tok_temp = stemmer.stem(tok_temp)
        # strip white space and append token to tokenised text array
        text_tokenised.append(tok_temp.strip())
        
    return text_tokenised



def build_model():
    """
    Build a ML pipeline using 
         - bag of words with TF-IDF;
         - Random Forest classifier;
         with hyperparameters tuned using grid search. 
         
    OUTPUTS:
        cv - instantiated pipelined model with optimal hyperparameters from grid search
    """
    
    # initiate pipeline with bag-of-words and TF-IDF transformation as feature representation;
    # followed by classification with a RandomForestClassifier.
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    # declare parameters for grid search
    parameters = {'clf__estimator__n_estimators': [40, 80],
                  'clf__estimator__min_samples_split': [2, 3, 4, 5],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    
    # grid search instantiation
    cv = GridSearchCV(pipeline, param_grid=parameters)
    print("Grid search object created.")
    
    return cv


def evaluate_model(model, X_test, Y_test, cat_names):
    """
    Evaluate and prints the model performance on test data based on the following metrics:
        - precision
        - recall
        - F1 score
    
    INPUTS:
        model - trained model
        X_test - test data features
        Y_test - labels of test data features for evaluation
        cat_names - names of the 36 categories
    
    """
    
    Y_pred = model.predict(X_test)
    
    # calculate the accuracy for each prediction
    for i in range(len(cat_names)):
        print("Category:", cat_names[i])
        print("Accuracy:", accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i]))
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print("\n")

def save_model(model, model_filepath):
    """
    Save trained model as a pickle file to a specified path.
    
    INPUTS:
        model - trained model.
        model_filepath - name and path to store pickle file of model being saved.
    """

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """
    Main executable function.
    No changes made to Udacity template, except for a script execution timer.
    """
    
    start_time = time.time()
    
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

    print("Script execution time: {} minutes |".format(round((time.time() - start_time)/60, 1)))

if __name__ == '__main__':
    main()