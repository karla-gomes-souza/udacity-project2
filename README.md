# Disaster Response Pipeline Project

Project 2 - Data Scientist Nanodegree (Udacity)

#### Table of Contents
1. Instructions
2. Project Motivation
3. File descriptions

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

#### Proejct Motivation
This project is a part of the Data Scientist Nanodegree from Udacity. The ideia is to apply the learned skills in the data from Appen, building a model for an API that classifies disaster messages. This model has to have processes such as ETL, ML pipeline and visualization with Flask.

#### File descriptions
There folder app constains the file with the flask web application, the models file contains the saved model and the data file constains the dataset.