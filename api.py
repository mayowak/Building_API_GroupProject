#!/usr/bin/env python3

# import relevant libraries
from flask import Flask, jsonify, request
import sys 
from joblib import dump, load
import traceback
import pandas as pd
import numpy as np

# api definition
app = Flask(__name__)

# define the predict function
@app.route('/predict', methods= ['POST']) # endpoint url will contain /predict

def predict():

    if loaded_model:
        try:

            # request the prediction file, which comes in a json format 
            json_file = request.json

            # convert the file to a dataframe
            query_df = pd.DataFrame(json_file)

            # get the dummy variables of the dataframe
            dummy_var = pd.get_dummies(query_df)

            # make sure the columns of the data to be predicted are in line with the columns of the trained dataset,
            # if the columns are smaller than expected, fill the excess columns with zeros
            dummy_df = dummy_var.reindex(columns = model_columns, fill_value = 0)

            # predict the survivors
            prediction = list(loaded_model.predict(dummy_df))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})

    else:

        print('Train the model first')

        return ('No model loaded')


# write the main function
if __name__ == '__main__':

    try:
        port = int(sys.argv[1]) # incase a command line port argument is specified use it as default port

    except:

        port = 12345 # if not use this


    loaded_model = load('model.pkl') # load model and assign to variable

    print('model loaded')

    model_columns = load('model_columns.pkl') # load model columns and assign to variable

    print('model columns loaded')

    app.run(port = port, debug = True)