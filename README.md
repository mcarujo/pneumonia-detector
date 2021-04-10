# Pneumonia Detector (X-Ray)
This project is the final project of my specialization in Data Science, Which used the CNN to classify chest X-ray image between healthy and pneumonia. The Model is provided in a web application where you can upload an X-Ray Image from a chest, and then receive the classification/analyzes. The whole project is a acamedic approach and should not be used by a docker or similiar person in a hospital for example.
  

## Dataset
![no text](https://i.imgur.com/jZqpV51.png)

 - Source: This dataset was taken from the Kaggle platform posted here in this [link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

#### Information
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.


# Approach

![Optional Text](./images/xray_cnn.png)

For this approach, I'm using CNN to identify the disease (Pneumonia). Before providing the image to the CNN must be processed to just provide in the image the chest part of the image. In the folder **models**, there are some Convolutional Neural Network that was tested to be used in the solution. It's used the image in Grayscale to feed and train the CNN.


In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery. For mor information click [here](https://en.wikipedia.org/wiki/Convolutional_neural_network).


X-rays are a type of radiation called electromagnetic waves. X-ray imaging creates pictures of the inside of your body. The images show the parts of your body in different shades of black and white. For more information click [here](https://en.wikipedia.org/wiki/X-ray).
<br /><br /><br />

# Application
It's was developed a web application to be possible select the X-ray images and predict them as well as train the model. You can select your X-ray image to apply and then the model will return the probability, class and explanation.
<br />

## Getting starting 

Can be used the Dockerfile to build the environment to run the application. Also is possible to pull the container from [Dockerhub](https://hub.docker.com/r/mamcarujo/pneumonia-detector).

<br /><br />

#### Building - Building the image in your machine.
```bash
docker build -t mamcarujo/pneumonia-detector .
```

<br /><br />

#### Pulling - Pulling the image in your machine.
```
docker pull mamcarujo/pneumonia-detector
```

<br /><br />

#### Running - Can be run the container using the following command.
```
docker run -p 80:5000 mamcarujo/pneumonia-detector
```


## User Interface
<br />
<img src="./images/update_image.png" width="50%">
<br />
<img src="./images/using_prediction.jpeg" width="50%">
