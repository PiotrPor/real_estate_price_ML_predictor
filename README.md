# real\_estate\_price\_ML\_predictor

AI model was created via Machine Learning to predict expected sell price of a real estate (house or apartment).



Dataset to train the model is read from SQLite database.
Data preparation is done using methods from numpy, pandas and sklearn packages
Neural Network (AI model) is created and then trained using TensorFlow Keras.

Streamlit is used to create a web app which serves as interface to interact with the trained AI model.



This repo contains 2 directories.
"ML\_model\_creation\_and\_training" contains:

* file with code used to create and train the neural network and process the training and test data
* datasets (tables saved in Excel files, their contents would be stored in a database)
* saved trained model (\*.keras file)
* Excel file with a table used to lookup label encoding

"STREAMLIT\_APP" directory contains the Streamlit web app project files

