# import libraries
import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine


# download necessary NLTK data
import nltk
nltk.download(['omw-1.4', 'punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath: str):
    """"
    Load data from database

    Args:
        database_filepath: string with database filepath  
    Returns:
        X: Series with features
        y: Pandas dataframe with targets
        target_names: list of target names
    """
    engine = create_engine("sqlite:///"+ database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df["message"]
    y = df.iloc[:, 4:]
    target_names = y.columns

    return X, y, target_names

def tokenize(text:str):
    """"
    Get tokens for sentences

    Args:
        text: str  
    Returns:
        clean_tokens: list with tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    """"
    Pipeline to train and optimize the ML model

    Args:
        None
    Returns:
        Optimized ML model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__min_samples_split': [2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2, verbose=2, cv=5)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """"
    Tests and prints the evaluation of the model

    Args:
        model: model file 
        X_test: Data with set of splitted test features
        Y_test: Data with splitted target points
        category_names: list of available categories
    Returns:
        Prints the evaluation metrics for the model
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)
    for category in category_names:
        print(category)
        print(classification_report(Y_test[category], Y_pred[category]))

def save_model(model, model_filepath):
    """
    Saves the model to pickle
    
    Args:
        model: model file
        model_filepath: model filepath
    
    Returns:
        None
    """
    pickle.dump(model, open("models/"+model_filepath, 'wb'))


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