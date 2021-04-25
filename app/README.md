# APP
Application to deploy the Pneumonia-Detector, here I tried to develop a project simple but effective in terms of training, prediction and monitoring. 


## which technologies/frameworks I used??
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [Tensorflow](https://www.tensorflow.org/) - [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Pandas](https://pandas.pydata.org/)


## Directory
- **data** store the datasets as CSV file.
- **model** store the model's files.
- **static** store static files that will be returned by the API.
- **temp** store temporary files.
- **templates** store the HTML files and CSS file used by the Flask.
- **app.py** the Flask file where contains the API configuration, routes, etc.
- **data_processing.py** the file contains the class DataProcessing responsible to process the dataset.
- **model.py** the file contains two classes ModelPredict and ModelTrain responsible to predict and train the CNN Model. 
- **test.py** the test file which contains unit-test for machine learning classes.
- **logs.txt** log file as format txt.
