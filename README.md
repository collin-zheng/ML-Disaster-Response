# Disaster Response Pipeline Project

### Project details:
This project is a predictive model that classifies messages related to disasters into 36 different categories.
It uses a random forest algorithm trained on roughly 20,000 messages. The training data and results are displayed visually.
Users are able to input their own disaster-related messages in a GUI interface and see classification results visually.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app:
    `python run.py`

3. In the browser, go to http://0.0.0.0:3001/.

### Known bugs:
Please note: In the original app/templates/master.html I replaced the line 
<script src="https://d14fo0winaifog.cloudfront.net/plotly-basic.js"></script> with 
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script> in order to get the visualisations to load. Apparently there is a plotly server availability issue with the old script, resulting in the histograms failing to load.
