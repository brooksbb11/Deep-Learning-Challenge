# Deep-Learning-Challenge
University of Arizona BootCamp 2023

# Background 
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Neural Network Model Report

# Overview
This project aims to develop a binary classifier that can predict the likelihood of applicants achieving success if they receive funding from Alphabet Soup. The project will utilize the features present in the given dataset and employ diverse machine learning methods to train and assess the model's performance. The objective is to optimize the model in order to attain an accuracy score surpassing 75%.

# Results
Data Preprocessing

The model aims to predict the success of applicants if they were to receive funding. This is indicated by the IS_SUCCESSFUL column in the dataset which is the target variable of the model.

 The feature variables are every column other than the target variable and the non-relevant variables such as EIN and NAMES. The features capture relevant information about the data and can be used in predicting the target variables, the non-relevant variables that are neither targets nor features will be drop from the dataset to avoid potential problems that might confuse the model.

During preprocessing, I implemented binning/bucketing for rare occurrences in the APPLICATION_TYPE and CLASSIFICATION columns. Subsequently, I transformed categorical data into numeric data using the get dummies encoding technique. I split the data into separate sets for features and targets, as well as for training and testing. Lastly, I scaled the data to ensure uniformity in the data distribution.

Compiling, Training, and Evaluating the Model

For the intital model, I included 3 layers: an input layer with 80 neurons, a second layer with 30 neurons, and an output layer with 1 neuron.I selected the relu activation function for the first and second layers, and the sigmoid activation function for the output layer since the goal was binary classification. To start, I had to train the model for 100 epochs and achieved an accuracy score of approximately 73.9% for the training data and 72.9% for the testing data. 

For the first optimization attempt I attempted to optimize the model's performance by first modifiying the number of layers and inputs to see if that would help increase the models accuracy. However adding more layers and inputs gave the same results. 73.9% on the training data and 72.9% on the testing data. 

For the second optimization attempt I to use the hyperparameter option. I used the Kerastuner to decide which activiation action to use, and decide the number of hidden layers adn neurons to be added. With this model the trained data scored a 72.8% accuracy and the testing data scored a 72.7%. This model was close to the other models, but is not the most accurate to use. 

For the third optimization attempt I dropped the STATUS column from the dataframe as well as the orginal two NAME and EIN. I dropped this column to see if that would help level the data since each value in that column was the same. Then for the model I went back to the orginal model to make changes and see if that would help. For the first and second layer instead of using relu I used tanh instead. Then for the third layer I used sigmoid. With this test the training data came out to 74.1% accuracy and 72.8% accuracy on the testing data. 


# Summary 
I was not able to attain the targeted accuracy of 75%. I would not suggest any of the models above since none of them hit 75%. I would say that the last model was the closest accuracy out of all the models, but all of my models are very similar. I do strongly believe that if I continue to make changes to the layers and neurons I could get to the goal of 75% accuracy. 

