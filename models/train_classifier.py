import sys
import re
import nltk
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath to the data base with clean data
    
    OUTPUT:
    X - pandas DataFrame with model input data
    Y - pandas DataFrame with model response data
    column names - column names for response data
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse_DataFrame', engine)
    X = df['message']
    Y = df[list(df.columns)[4:]]
    return X, Y, Y.columns


def tokenize(text):
    '''
    INPUT:
    text - text extracted from a database row
    
    OUTPUT:
    tokens - list with all text tokens
    '''
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # stop words
    stop_words = stopwords.words("english")

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''    
    OUTPUT:
    model - a model which has been optimized for the global task by using GridSearchCV
    '''
    # build pipeline
    forest = RandomForestClassifier(n_estimators = 50, random_state = 1) 
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            #('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(forest))
    ])
    parameters = {
            'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            'features__text_pipeline__vect__max_df': (0.5, 1.0),
            'clf__estimator__n_estimators': [50, 200],
            'clf__estimator__min_samples_split': [2, 4],
            'features__transformer_weights': (
                {'text_pipeline': 1, 'starting_verb': 0.5},
                {'text_pipeline': 0.8, 'starting_verb': 1})
    }
    
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, verbose = 2)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - model to be evaluated
    X_test - test input data for evaluation
    Y_test - test response data for evaluation
    category_names - column names for response data
    '''
    for i, column in enumerate(category_names):
        print('Test results for "' + column + '" :\n')
        print(classification_report(Y_test[column], pd.DataFrame(model.predict(X_test))[i]))
        print('\n\n')


def save_model(model, model_filepath):
    '''
    INPUT:
    model - model to be saved as pkl
    model_filepath - filepath and model file name 
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
        print(model.best_params_)
        
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