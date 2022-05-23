#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, render_template, request , jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
import deployment_inputs as get_input
import logging


# In[3]:

logging.basicConfig(filename='deployment_logs.log', level=logging.INFO,
                   format='%(levelname)s:%(asctime)s:%(message)s') 


app = Flask(__name__,template_folder='Template')
model = pickle.load(open('Final_Model.pkl', 'rb'))


# In[ ]:


@app.route('/',methods=['GET'])


# In[ ]:


def Home():
    return render_template('index.html')


# In[ ]:


@app.route("/predict", methods=['POST'])


# In[ ]:


def predict():
    
    if request.method == 'POST':
        
        
        logging.info('Active User Found')
        
        
        inputs={'LIMIT_BAL': float(request.form['LIMIT_BAL']),'SEX': request.form['SEX'],'EDUCATION': request.form['EDUCATION'],
'MARRIAGE': request.form['MARRIAGE'],'AGE': int(request.form['AGE']),'PAY_0': request.form['REPAYMENT_STATUS_SEPT'],'PAY_2': request.form['REPAYMENT_STATUS_AUGUST'],
'PAY_3': request.form['REPAYMENT_STATUS_JULY'],'PAY_4': request.form['REPAYMENT_STATUS_JUNE'],'PAY_5': request.form['REPAYMENT_STATUS_MAY'],
'PAY_6': request.form['REPAYMENT_STATUS_APRIL'],'BILL_AMT1':float(request.form['BILL_AMT_SEPT']),'BILL_AMT2':float(request.form['BILL_AMT_AUGUST']),
'BILL_AMT3':float(request.form['BILL_AMT_JULY']),'BILL_AMT4':float(request.form['BILL_AMT_JUNE']),'BILL_AMT5':float(request.form['BILL_AMT_MAY']),
'BILL_AMT6':float(request.form['BILL_AMT_APRIL']),'PAY_AMT1': float(request.form['PAY_AMT_SEPT']),'PAY_AMT2': float(request.form['PAY_AMT_AUGUST']),
'PAY_AMT3': float(request.form['PAY_AMT_JULY']),'PAY_AMT4': float(request.form['PAY_AMT_JUNE']),'PAY_AMT5': float(request.form['PAY_AMT_MAY']),
'PAY_AMT6': float(request.form['PAY_AMT_APRIL'])}
        
        logging.info(f"Inputs from user {inputs}")
        
        inputs=get_input.get_inputs_for_model_tme(input_dict=inputs)


        prediction=model.predict_proba([inputs])[0][1]
        
        
        logging.info(f"Probability Of Default is {prediction}")
        
        
    
        return render_template('index.html',prediction_text="Probability Of Default Is== {}".format(prediction))
    
    else:
        return render_template('index.html')


# In[1]:


if __name__=="__main__":
    app.run(debug=True)

