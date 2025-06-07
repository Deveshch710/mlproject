#this is mainly for deployment purpose as we are using AWS Elastic Beanstalk
# and we need to run the application using the application variable

from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline


#entry point of the application
application = Flask(__name__)

app= application

#route for the home page
@app.route('/')
def home():
    return render_template('index.html')

#rote for the prediction page
@app.route('/predictdata', methods=['GET','POST']) 
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
             gender=request.form.get('gender'),
             race_ethnicity=request.form.get('race_ethnicity'),
             parental_level_of_education=request.form.get('parental_level_of_education'),
             lunch=request.form.get('lunch'),
             test_preparation_course=request.form.get('test_preparation_course'),
             reading_score=float(request.form.get('reading_score')),
             writing_score=float(request.form.get('writing_score')),
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictionPipeline()
        results=predict_pipeline.predict(pred_df)
        
        return render_template('home.html',results=results[0])


#run the application
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)