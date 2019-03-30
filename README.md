# Disaster Response Pipeline Project

### Project Overview
Goal of this project is to design a web app that classifies a given twitter message into one or more of 36 cathegories. 

The work consists of following  steps:
1. ETL Pipeline: loads the training data, transforsms, cleans and saves it into an SQL database.
2. ML Pipeline: loads the data after ETL step, defines a machine learning pipeline, fits it to the data and save resulting model in pkl file
3. Web app: uses a ML Model created by ML Pipeline to predict a category of given tweet message.

### Installations

In order to run this project following python libraries are needed: 
    - ETL Pipeline: pandas, numpy, langid, sqlalchemy
    - ML Pipeline: pyndas, numpy, sqlite3, re, time, warnings, pickle, nltk, matplotlib, seaborn, sqlalchemy, sklearn, scipy
    - Web app: pandas, json, plotly, nltk, flask, sklearn, sqlalchemy
   
### Files
This project contains two jupyter notebooks: **ETL_Pipeline_Preparation.ipynb** and **ML_Pipeline_Preparation.ipynb** both were used to prepare ETL and ML pipelines described above. The actual ETL and ML pipelines are stored in  **data/process_data.py** and **models/train_classifier.py** respectively. Source (raw) data that is used by ETL pipeline is stored in two csv files: **data/disaster_categories.csv** and **data/disaster_messages.csv**. The SQL databased used by ML pipeline is stored into a **twitter_messages.db** file. Finally, a web app can be found under **app/run.py**

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to **localhost:3001** 
