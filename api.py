#!/usr/bin/env python3

# import relevant libraries
from flask import Flask, jsonify, request, render_template
import json
import sys 
from joblib import dump, load
import traceback
import pandas as pd
import numpy as np

# api definition
app = Flask(__name__)

# display the home page index.html
@app.route('/')
def display_form():
    return render_template('./index.html')
    

# define the predict function
@app.route('/predict', methods= ['POST']) # endpoint url will contain /predict
def predict():

    # if the predict button is clicked run this if block
    if request.form.get("predict") == "predict":
        # execute this try block if the predict button is clicked
        try:

            loaded_model = load('model.pkl') # load model and assign to variable

            # get the form values by referencing the `name` attribute for each input in the form
            pclass = request.form.get('Pclass')
            sex = request.form.get('Sex_male')
            age = request.form.get('Age')
            fare = request.form.get('Fare')
            embarked = request.form.get('Embarked_Q')
            embarked2 = request.form.get('Embarked_S')

            
            # get all input values from form as a list of tuples
            data_list = [
                        (pclass), (sex),
                        (age), (fare),
                        (embarked), (embarked2)
                        ]

            # convert list of tuples to an array of appropriate shape and then to a dataframe
            query_df = pd.DataFrame(np.array(data_list).reshape(1, -1))

            # get the dummy variables of the dataframe
            dummy_var = pd.get_dummies(query_df)

            # display survived if the predicted value is 1 else display died
            prediction = "".join(['Survived' if loaded_model.predict(dummy_var.values) == 1 else 'Died'])

            # return result.html which holds the prediction value
            return render_template("result.html" , prediction = prediction)

        except: # if model is not loaded, return the strings below

            return ("<h2> No model Loaded! <br><br> Please Train Model First... </h2>")

    # else if the train button is clicked
    elif request.form.get("train") == "train":

        # use the exec function to run a python script
        exec(open("./model.py").read())

        return "<h2> Model Trained! </h2>"

        
# function that prints the head of the cleaned dataset
@app.route('/view_data', methods = ['POST'])
def get_head_tail_info():

    # get the cleaned dataset
    read_file = pd.read_csv('./dataset.csv')

    # get the form submit button whose name is head and value is head
    if request.form.get("head") == "head":
        # show just the head
        return read_file.head().to_html()

    # get the form submit button whose name is tail and value is tail
    elif request.form.get("tail") == "tail":
        # show just the tail
        return read_file.tail().to_html()

    # get the form submit button whose name is info and value is info
    elif request.form.get('info') == "info":
        # return the dataset description
        return read_file.describe().to_html()

# write the main function
if __name__ == '__main__':

    try:
        port = int(sys.argv[1]) # incase a command line port argument is specified use it as default port

    except:

        port = 13579 # if not use this


    app.run(port = port, debug = True)