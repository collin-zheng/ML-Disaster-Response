# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app:
    `python run.py`

3. In the browser, go to http://0.0.0.0:3001/.

Please note: In the original app/templates/master.html I replaced the line 
<script src="https://d14fo0winaifog.cloudfront.net/plotly-basic.js"></script> with 
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script> in order to get the visualisations to load.
It took me ages to work out why the visualisations wouldn't load!!! Thanks to a Udacity mentor for offering this fix. Apparently there is a plotly server availability issue with the old script.
In my zip file I've included the amended master.html file in case it is useful for the evaluator.