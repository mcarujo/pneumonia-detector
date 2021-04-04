# APP
Application to deploy a time series forecast.

## Introduction
Here I tried to develop a project simple but effective in terms of training, prediction and monitoring. The model trains with the data already in CSV, which means that I'm expecting an ETL that get the data from a raw format and then transform it into CSV, this process was made by me in the notebooks. For predictions I'm expecting how much days ahead the user would like to see. In the end, will be provided with a screen to analyze the model performance and historical logs.


## which technologies/frameworks I used??
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [Prophet](https://facebook.github.io/prophet/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/)


## Directory
- **data** store the datasets as CSV file.
- **model** store the model's joblib file.
- **templates** store the HTML files and CSS file used by the Flask.
- **app.py** the Flask file where contains the API configuration, routes, etc.
- **data_processing.py** the file contains the class DataProcessing responsible to process the dataset.
- **model.py** the file contains to classes ModelPredict and ModelTrain responsible to predict and train the Prophet model. 
- **test.py** the test file which contains unit-test for machine learning classes.
- **logs.txt** log file as format txt.
