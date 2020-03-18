# Disaster Response Pipeline Project

## Project Description

Utilizing and analyzing disaster response data from Figure Eight to build a model that classifies disaster messages and display results in a Flask web application.

## Application File Layout

    .
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   ├── process_data.py                  # Data cleaning
    │   └── DisasterResponse.db              # Database for Disaster Response data
    ├── models
    │   ├── train_classifier.py              # Train ML model      
    │   └── classifier.pkl                   # Pickle file of model     
    ├── README.md
    └── requirements.txt

## Instructions (Skip steps 2 and 3 if .pkl file and .db database exists):
1. Run the following command to install necessary libraries.
    `pip install -r requirements.txt`

2. Run the following commands in the project's root directory to create your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:33507/

## Example
Type in: 
```
We need supplies and food after the tornado destroyed critical facilities
```
and click `Classify Message`

![Example](ex.png)
